"""
Multi-Source Best-Effort Adaptation (MSBEA) diffusion policy.

Extends BEA to M source domains by factoring the per-sample weight into a
source-domain weight w_s and a within-source sample weight q̃_i:

    q_eff_i = w_{s(i)} · q̃_i      for i ∈ ∪_s S_s
    q_eff_j = q^T_j                 for j ∈ T

The combined dataset is indexed as
    [S_1 samples | S_2 samples | … | S_M samples | T samples]
so that indices 0..m_total-1 are source and m_total..m_total+n-1 are target,
exactly matching the BEA index layout.

The training loop alternates three phases per epoch:
  1. Update discrepancy d  (optional, inherited from BEA)
  2. Update w, q̃, q^T     (minibatch projected subgradient)
  3. Update h              (q_eff-weighted ERM)

Config layout (under algo.msbea, on top of inherited algo.bea):
  rho_1, rho_2              — regularization on w
  w.eta                     — step size for w subgradient
  w.min, w.max              — box bounds on each w_s
  q.eta_T                   — step size for target q update
"""

from collections import OrderedDict

import torch

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy_bea import BEADiffusionPolicyUNet


@register_algo_factory_func("diffusion_policy_msbea")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return MSBEADiffusionPolicyUNet, {}
    else:
        raise RuntimeError("MSBEA currently only supports UNet backbone")


class MSBEADiffusionPolicyUNet(BEADiffusionPolicyUNet):
    """
    Multi-Source BEA Diffusion Policy with UNet backbone.

    Inherits _diffusion_forward, process_batch_for_training,
    compute_losses_for_q_update, _extract_obs_embeddings, _get_gamma,
    compute_r_dp from BEADiffusionPolicyUNet.

    Overrides all weight-management, training, and serialization logic.
    """

    def _create_networks(self):
        super()._create_networks()
        # MSBEA-specific state (initialized in init_msbea_weights)
        self.w = None             # (M,)  source domain weights
        self.w0 = None            # (M,)  reference domain weights
        self.q_tilde = None       # (m_total,)  within-source sample weights
        self.q_T = None           # (n,)  target sample weights
        self.domain_id = None     # (m_total,) long — source domain index
        self.source_sizes = None  # list[int]
        self.m_total = 0
        # self.m, self.n, self.p0, self.d are inherited from BEA and reused
        self._msbea_initialized = False

    #  Weight initialization

    def init_msbea_weights(self, source_sizes, n):
        """
        Initialize the MSBEA weight structure.

        Args:
            source_sizes: list of ints — number of samples per source domain.
            n: int — number of target samples.
        """
        bea_cfg = self.algo_config.bea
        ms_cfg = self.algo_config.msbea

        M = len(source_sizes)
        m_total = sum(source_sizes)
        total = m_total + n

        self.source_sizes = list(source_sizes)
        self.m_total = m_total
        # Set inherited BEA m/n so that parent helpers (update_d_knn, etc.)
        # correctly split at the source/target boundary.
        self.m = m_total
        self.n = n

        # Domain-id tensor: maps each source sample to its domain 0..M-1
        ids = []
        for s, sz in enumerate(source_sizes):
            ids.append(torch.full((sz,), s, dtype=torch.long))
        self.domain_id = torch.cat(ids)  # (m_total,)

        # Source domain weights w and reference w0
        self.w = torch.full((M,), 1.0 / M, dtype=torch.float32)
        self.w0 = torch.full((M,), 1.0 / M, dtype=torch.float32)

        # Reference per-sample weights p0 (same layout as BEA)
        p0 = torch.zeros(total, dtype=torch.float32)
        if getattr(bea_cfg, 'p0_source_uniform', False):
            p0[:m_total] = 1.0
        else:
            p0[:m_total] = bea_cfg.p0_source
        if bea_cfg.p0_target_uniform:
            p0[m_total:] = 1.0
        else:
            p0[m_total:] = bea_cfg.p0_target
        self.p0 = p0

        # Discrepancy penalties (source > 0, target = 0)
        d = torch.zeros(total, dtype=torch.float32)
        d[:m_total] = bea_cfg.d.value
        self.d = d

        # Within-source sample weights q̃ — initialize from p0 / w
        # so that q_eff = w * q_tilde ≈ p0 at init
        w_per_sample = self.w[self.domain_id]  # (m_total,)
        self.q_tilde = (p0[:m_total] / w_per_sample.clamp(min=1e-8)).clone()
        self.q_tilde.clamp_(0.0, bea_cfg.q.max)

        # Target sample weights q^T — initialize from p0
        self.q_T = p0[m_total:].clone()
        target_min = bea_cfg.q.target_min
        if target_min > 0 and n > 0:
            self.q_T.clamp_(min=target_min)

        # Loss EMA (optional)
        if bea_cfg.loss_ema_beta > 0:
            self.loss_ema = torch.zeros(total, dtype=torch.float32)
        else:
            self.loss_ema = None

        # Also set BEA's q to q_eff for compatibility with any residual
        # parent code paths (e.g. log_info checking q_source_mean).
        self.q = self.get_q_eff()

        self._bea_initialized = True
        self._msbea_initialized = True
        print(
            f"[MSBEA] Initialized: M={M}, source_sizes={source_sizes}, "
            f"n={n}, total={total}, w={self.w.tolist()}"
        )

    #  Effective weight computation

    def get_q_eff(self):
        """Assemble the effective per-sample weight vector (m_total + n,)."""
        w_per_sample = self.w[self.domain_id]  # (m_total,)
        q_eff = torch.zeros(self.m_total + self.n, dtype=torch.float32)
        q_eff[:self.m_total] = w_per_sample * self.q_tilde
        q_eff[self.m_total:] = self.q_T
        return q_eff

    #  w-update  (source domain weights)

    @torch.no_grad()
    def update_w(self, all_losses):
        """
        Subgradient step on source domain weights w using full-dataset losses.

        g_{w_s} = Σ_{i∈S_s} q̃_i (ℓ_i + λ_d d_i)
                + λ_1 Σ_{i∈S_s} q̃_i sign(w_s q̃_i − p0_i)
                + 2 λ_2 w_s Σ_{i∈S_s} q̃_i²
                + ρ_1 sign(w_s − w0_s)  +  2 ρ_2 w_s

        Then w ← Π_{Δ_M}(w − η_w · g_w).
        """
        bea_cfg = self.algo_config.bea
        ms_cfg = self.algo_config.msbea
        M = len(self.source_sizes)

        lam1 = bea_cfg.lambda_1
        lam2 = bea_cfg.lambda_2
        lam_d = bea_cfg.d.lambda_d
        rho1 = ms_cfg.rho_1
        rho2 = ms_cfg.rho_2

        src_losses = all_losses[:self.m_total].cpu()
        src_d = self.d[:self.m_total]
        src_p0 = self.p0[:self.m_total]

        # Scale eta by 1/m_avg so the step size is independent of dataset
        # size while keeping the mathematically correct sum-form gradient.
        m_avg = float(self.m_total) / M
        eta_w = ms_cfg.w.eta / m_avg

        grad_w = torch.zeros(M, dtype=torch.float32)
        for s in range(M):
            mask = self.domain_id == s
            qt_s = self.q_tilde[mask]
            l_s = src_losses[mask]
            d_s = src_d[mask]
            p0_s = src_p0[mask]
            w_s = self.w[s]

            cost = l_s + lam_d * d_s  # (m_s,)
            q_eff_s = w_s * qt_s

            grad_w[s] = (
                (qt_s * cost).sum()
                + lam1 * (qt_s * torch.sign(q_eff_s - p0_s)).sum()
                + 2.0 * lam2 * w_s * (qt_s ** 2).sum()
                + rho1 * torch.sign(w_s - self.w0[s])
                + 2.0 * rho2 * w_s
            )

        self._last_w_grad = grad_w.clone()
        self._last_w_eta = eta_w
        self.w = self.w - eta_w * grad_w
        self._project_w_simplex()

    def _project_w_simplex(self):
        """Project w onto the probability simplex {w >= w_min, sum(w) = 1}."""
        ms_cfg = self.algo_config.msbea
        w = self.w.clamp(min=ms_cfg.w.min, max=ms_cfg.w.max)

        # Euclidean projection onto simplex via sorting
        M = w.shape[0]
        sorted_w, _ = torch.sort(w, descending=True)
        cumsum = torch.cumsum(sorted_w, dim=0)
        rho_candidates = sorted_w - (cumsum - 1.0) / torch.arange(1, M + 1, dtype=torch.float32)
        rho_idx = (rho_candidates > 0).long().sum() - 1
        theta = (cumsum[rho_idx] - 1.0) / float(rho_idx + 1)
        w = (w - theta).clamp(min=ms_cfg.w.min)

        # Renormalize to guarantee exact sum = 1 after clamping
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = torch.full_like(w, 1.0 / M)

        self.w = w

    #  q-update  (per-sample weights for source and target)

    @torch.no_grad()
    def update_q_msbea(self, batch_indices, per_sample_losses):
        """
        Projected subgradient step on q̃ (source) and q^T (target).

        Source (i < m_total):
            g_i = w_s (ℓ_i + λ_d d_i) + λ_1 w_s sign(w_s q̃_i − p0_i) + 2 λ_2 w_s² q̃_i

        Target (j >= m_total):
            g_j = ℓ_j + λ_1 sign(q^T_j − p0_j) + 2 λ_2 q^T_j
        """
        bea_cfg = self.algo_config.bea
        ms_cfg = self.algo_config.msbea

        idx = batch_indices.cpu()
        losses = per_sample_losses.detach().cpu()

        lam1 = bea_cfg.lambda_1
        lam2 = bea_cfg.lambda_2
        lam_d = bea_cfg.d.lambda_d
        eta_q = bea_cfg.q.eta
        eta_T = ms_cfg.q.eta_T
        q_max = bea_cfg.q.max
        target_min = bea_cfg.q.target_min

        # Optional loss EMA
        if self.loss_ema is not None:
            beta = bea_cfg.loss_ema_beta
            self.loss_ema[idx] = beta * self.loss_ema[idx] + (1.0 - beta) * losses
            losses = self.loss_ema[idx]

        src_mask = idx < self.m_total
        tgt_mask = ~src_mask

        # Source samples
        if src_mask.any():
            src_idx = idx[src_mask]       # global indices in [0, m_total)
            src_l = losses[src_mask]
            src_d = self.d[src_idx]
            src_p0 = self.p0[src_idx]

            dom_ids = self.domain_id[src_idx]
            w_s = self.w[dom_ids]         # per-sample w for this batch
            qt = self.q_tilde[src_idx]
            q_eff = w_s * qt

            grad = (
                w_s * (src_l + lam_d * src_d)
                + lam1 * w_s * torch.sign(q_eff - src_p0)
                + 2.0 * lam2 * (w_s ** 2) * qt
            )
            qt = qt - eta_q * grad
            qt = qt.clamp(0.0, q_max)
            self.q_tilde[src_idx] = qt

        # Target samples
        if tgt_mask.any():
            tgt_global_idx = idx[tgt_mask]          # global indices in [m_total, m_total+n)
            tgt_local_idx = tgt_global_idx - self.m_total
            tgt_l = losses[tgt_mask]
            tgt_p0 = self.p0[tgt_global_idx]
            qt_T = self.q_T[tgt_local_idx]

            grad = (
                tgt_l
                + lam1 * torch.sign(qt_T - tgt_p0)
                + 2.0 * lam2 * qt_T
            )
            qt_T = qt_T - eta_T * grad
            qt_T = qt_T.clamp(0.0, q_max)
            if target_min > 0:
                qt_T = qt_T.clamp(min=target_min)
            self.q_T[tgt_local_idx] = qt_T

    def project_q_simplex(self):
        """
        Project q_eff jointly over source + target, matching BEA's
        single-constraint projection: sum(q_eff) = n + alpha * m_total.

        After projection, recover q̃ and q^T from the projected q_eff.
        """
        bea_cfg = self.algo_config.bea
        q_max = bea_cfg.q.max
        target_min = bea_cfg.q.target_min

        if bea_cfg.q.sum_constraint_enabled:
            alpha = bea_cfg.q.sum_constraint_alpha
            target_sum = float(self.n) + alpha * float(self.m_total)

            w_per_sample = self.w[self.domain_id]  # (m_total,)
            q_eff = torch.cat([w_per_sample * self.q_tilde, self.q_T])

            # Element-wise bounds: source q_eff ∈ [0, w_s * q_max],
            # target q_eff ∈ [target_min, q_max]
            q_lo = torch.zeros(self.m_total + self.n, dtype=torch.float32)
            if target_min > 0 and self.n > 0:
                q_lo[self.m_total:] = target_min

            q_hi = torch.cat([
                w_per_sample * q_max,
                torch.full((self.n,), q_max, dtype=torch.float32),
            ])

            q_eff = self._bisection_project(q_eff, target_sum, q_lo, q_hi)

            # Recover q̃ and q^T from projected q_eff
            self.q_tilde = q_eff[:self.m_total] / w_per_sample.clamp(min=1e-8)
            self.q_T = q_eff[self.m_total:]
        else:
            self.q_tilde.clamp_(0.0, q_max)
            if target_min > 0 and self.n > 0:
                self.q_T.clamp_(min=target_min)

        # Rebuild q for any code that reads self.q
        self.q = self.get_q_eff()

    @staticmethod
    def _bisection_project(q, target_sum, lo, hi, max_iter=64, tol=1e-6):
        """Bisection projection onto {q : lo_i <= q_i <= hi_i, sum(q) = target_sum}.

        lo/hi can be scalars or tensors for element-wise bounds.
        """
        if isinstance(lo, (int, float)):
            q_lo = torch.full_like(q, lo)
        else:
            q_lo = lo
        if isinstance(hi, (int, float)):
            q_hi = torch.full_like(q, hi)
        else:
            q_hi = hi

        min_possible = q_lo.sum().item()
        max_possible = q_hi.sum().item()
        if target_sum < min_possible or target_sum > max_possible:
            target_sum = max(min_possible, min(max_possible, target_sum))

        lo_nu = float(q.min()) - float(q_hi.max()) - 1.0
        hi_nu = float(q.max()) + 1.0

        for _ in range(max_iter):
            mid = (lo_nu + hi_nu) / 2.0
            projected = torch.clamp(q - mid, min=q_lo, max=q_hi)
            current_sum = projected.sum().item()
            if current_sum > target_sum:
                lo_nu = mid
            else:
                hi_nu = mid
            if abs(current_sum - target_sum) < tol:
                break

        nu = (lo_nu + hi_nu) / 2.0
        return torch.clamp(q - nu, min=q_lo, max=q_hi)

    #  q_eff-weighted training step

    def train_on_batch(self, batch, epoch, validate=False):
        """
        q_eff-weighted training step.

        Uses the effective weight q_eff = w * q̃ (source) or q^T (target)
        looked up per sample via the global index.
        """
        assert self._msbea_initialized, (
            "MSBEA weights not initialized. Call model.init_msbea_weights() "
            "before training."
        )

        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = OrderedDict()
            assert validate or self.nets.training

            per_sample_loss, _ = self._diffusion_forward(batch)

            indices = batch["index"].long()
            q_eff = self.get_q_eff()
            q_batch = q_eff[indices.cpu()].to(self.device).detach()

            weighted_loss = (q_batch * per_sample_loss).mean()
            unweighted_loss = per_sample_loss.mean()

            losses = {
                "l2_loss": unweighted_loss,
                "weighted_loss": weighted_loss,
                "q_eff_mean": q_batch.mean(),
                "q_eff_std": q_batch.std() if B > 1 else torch.tensor(0.0),
            }

            # Per-domain stats
            src_mask = indices < self.m_total
            tgt_mask = ~src_mask
            if src_mask.any():
                losses["q_eff_source_mean"] = q_batch[src_mask].mean()
                src_idx = indices[src_mask].cpu()
                dom_ids = self.domain_id[src_idx]
                for s in range(len(self.source_sizes)):
                    s_mask = dom_ids == s
                    if s_mask.any():
                        losses[f"q_eff_src{s}_mean"] = q_batch[src_mask][s_mask].mean()
            if tgt_mask.any():
                losses["q_eff_target_mean"] = q_batch[tgt_mask].mean()

            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=weighted_loss,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)

                info["policy_grad_norms"] = policy_grad_norms

        return info

    #  Logging

    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["l2_loss"].item()
        log["Weighted_Loss"] = info["losses"]["weighted_loss"].item()
        log["q_eff_mean"] = info["losses"]["q_eff_mean"].item()
        log["q_eff_std"] = info["losses"]["q_eff_std"].item()
        if "q_eff_source_mean" in info["losses"]:
            log["q_eff_source_mean"] = info["losses"]["q_eff_source_mean"].item()
        if "q_eff_target_mean" in info["losses"]:
            log["q_eff_target_mean"] = info["losses"]["q_eff_target_mean"].item()
        for s in range(len(self.source_sizes)):
            key = f"q_eff_src{s}_mean"
            if key in info["losses"]:
                log[key] = info["losses"][key].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    #  Serialization

    def serialize(self):
        # Call grandparent (DiffusionPolicyUNet) to avoid BEA's serialize
        # writing stale q/p0 under the "bea" key.
        from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
        d = DiffusionPolicyUNet.serialize(self)
        d["msbea"] = {
            "w": self.w,
            "w0": self.w0,
            "q_tilde": self.q_tilde,
            "q_T": self.q_T,
            "p0": self.p0,
            "d": self.d,
            "domain_id": self.domain_id,
            "source_sizes": self.source_sizes,
            "m_total": self.m_total,
            "n": self.n,
            "loss_ema": self.loss_ema,
        }
        return d

    def deserialize(self, model_dict, load_optimizers=False):
        from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
        DiffusionPolicyUNet.deserialize(self, model_dict, load_optimizers=load_optimizers)
        if "msbea" in model_dict:
            ms = model_dict["msbea"]
            self.w = ms["w"]
            self.w0 = ms["w0"]
            self.q_tilde = ms["q_tilde"]
            self.q_T = ms["q_T"]
            self.p0 = ms["p0"]
            self.d = ms["d"]
            self.domain_id = ms["domain_id"]
            self.source_sizes = ms["source_sizes"]
            self.m_total = ms["m_total"]
            self.m = ms["m_total"]
            self.n = ms["n"]
            self.loss_ema = ms.get("loss_ema", None)
            self.q = self.get_q_eff()
            self._bea_initialized = True
            self._msbea_initialized = True
