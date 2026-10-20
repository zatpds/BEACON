"""
Best-Effort Adaptation (BEA) diffusion policy.

Co-trains the UNet diffusion policy with per-sample weights q over the combined
source+target dataset (q-weighted ERM). The combined dataset U = S ∪ T is
reindexed so that indices 0..m-1 are source and m..m+n-1 are target.  The weight
vector q ∈ R^{m+n} lives on CPU and is sliced to GPU per mini-batch via the
"index" field that MetaDataset already provides in each sample.

Config layout (under algo.bea):
  lambda_1, lambda_2          — regularization coefficients
  q.mode                      — "full" or "minibatch"
  q.update_freq               — update q every K epochs
  q.max                       — upper bound on each q_i
  q.target_min                — lower bound for target q_i
  q.eta                       — step size (minibatch mode only)
  q.sum_constraint_enabled    — enforce sum(q) = target_sum
  q.sum_constraint_alpha      — target_sum = n + alpha * m
  d.mode                      — "d_hat", "knn", or "classifier"
  d.k                         — k-NN k (knn mode only)
  d.value                     — constant d (d_hat mode only)
  d.lambda_d                  — coefficient on discrepancy penalty
  p0_source, p0_target, p0_target_uniform — reference weights
  loss_ema_beta               — EMA smoothing (0 = disabled)
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


@register_algo_factory_func("diffusion_policy_bea")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return BEADiffusionPolicyUNet, {}
    else:
        raise RuntimeError("BEA currently only supports UNet backbone")


class BEADiffusionPolicyUNet(DiffusionPolicyUNet):
    """BEA (q-weighted ERM) co-training on top of the UNet diffusion policy."""

    def _create_networks(self):
        super()._create_networks()
        self.q = None        # (m+n,) CPU float tensor
        self.p0 = None       # (m+n,) reference weights
        self.d = None         # (m+n,) discrepancy penalties
        self.m = 0            # number of source samples
        self.n = 0            # number of target samples
        self.loss_ema = None  # (m+n,) EMA of per-sample losses (None if disabled)
        self._bea_initialized = False

    # BEA weight initialization

    def init_bea_weights(self, m, n):
        """
        Initialize per-sample weights q, reference weights p0, and
        discrepancy penalties d for the combined dataset of size m+n.

        Must be called after the model is created and before training starts.

        Args:
            m (int): number of source samples (indices 0..m-1)
            n (int): number of target samples (indices m..m+n-1)
        """
        cfg = self.algo_config.bea
        self.m = m
        self.n = n
        total = m + n

        # Reference weights p0
        p0 = torch.zeros(total, dtype=torch.float32)
        if getattr(cfg, 'p0_source_uniform', False):
            p0[:m] = 1.0
        else:
            p0[:m] = cfg.p0_source
        if cfg.p0_target_uniform:
            p0[m:] = 1.0
        else:
            p0[m:] = cfg.p0_target
        self.p0 = p0

        # Discrepancy penalties: d_hat > 0 for source, 0 for target
        d = torch.zeros(total, dtype=torch.float32)
        d[:m] = cfg.d.value
        self.d = d

        # Initialize q = p0 (algorithm line 3)
        self.q = p0.clone()

        # Loss EMA buffer for variance reduction (optional)
        if cfg.loss_ema_beta > 0:
            self.loss_ema = torch.zeros(total, dtype=torch.float32)
        else:
            self.loss_ema = None

        self._bea_initialized = True
        ema_str = f"beta={cfg.loss_ema_beta:.3f}" if self.loss_ema is not None else "off"
        print(
            f"[BEA] Initialized weights: m={m}, n={n}, q_shape=({total},), "
            f"loss_ema={ema_str}, q.target_min={cfg.q.target_min}"
        )

    # Weight-decay helpers (gamma, R_DP) for the ||q||_inf * R_DP(h) term

    def _get_gamma(self):
        """Return the weight-decay coefficient (L2 regularization) from the optimizer config."""
        return float(self.optim_params["policy"]["regularization"]["L2"])

    @torch.no_grad()
    def compute_r_dp(self):
        """Compute R_DP(h) = 0.5 * sum_w ||w||^2 over all model parameters."""
        total = 0.0
        for p in self.nets.parameters():
            total = total + p.data.pow(2).sum()
        return 0.5 * total

    # Dynamic discrepancy (k-NN & classifer)

    @torch.no_grad()
    def _extract_obs_embeddings(self, data_loader, obs_normalization_stats=None):
        """
        Single forward pass through the obs encoder to collect (index, embedding)
        for every sample in data_loader.  Returns tensors sorted by index.
        """
        all_idx, all_emb = [], []
        self.set_eval()
        for batch in data_loader:
            input_batch = self.process_batch_for_training(batch)
            input_batch = self.postprocess_batch_for_training(
                input_batch, obs_normalization_stats=obs_normalization_stats,
            )
            inputs = {"obs": input_batch["obs"], "goal": input_batch["goal_obs"]}
            obs_features = TensorUtils.time_distributed(
                inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True,
            )
            emb = obs_features.flatten(start_dim=1)  # (B, D)
            all_idx.append(input_batch["index"].long().cpu())
            all_emb.append(emb.cpu())

        all_idx = torch.cat(all_idx)
        all_emb = torch.cat(all_emb)
        order = all_idx.argsort()
        return all_idx[order], all_emb[order]

    @torch.no_grad()
    def update_d_knn(self, data_loader, obs_normalization_stats=None):
        """
        Recompute per-source discrepancy d_i as mean k-NN distance to target
        embeddings in obs-encoder feature space, normalised by mean
        within-target k-NN distance so that d_i ≈ 1 means "fits in with
        target" and d_i >> 1 means "outlier relative to target spread".
        """
        cfg = self.algo_config.bea
        k = cfg.d.k

        indices, embeddings = self._extract_obs_embeddings(
            data_loader, obs_normalization_stats,
        )

        src_emb = embeddings[:self.m]   # (m, D)
        tgt_emb = embeddings[self.m:]   # (n, D)
        n = tgt_emb.shape[0]

        effective_k = min(k, n)

        # Source-to-target k-NN distances
        st_dists = torch.cdist(src_emb, tgt_emb, p=2)       # (m, n)
        st_topk, _ = st_dists.topk(effective_k, dim=1, largest=False)
        raw_d = st_topk.mean(dim=1)  # (m,)

        # Within-target k-NN distances (use k+1 since self is included,
        # then drop the zero self-distance)
        tt_dists = torch.cdist(tgt_emb, tgt_emb, p=2)       # (n, n)
        tt_k = min(effective_k + 1, n)
        tt_topk, _ = tt_dists.topk(tt_k, dim=1, largest=False)
        # Column 0 is self-distance (0), take columns 1..tt_k
        tgt_ref = tt_topk[:, 1:].mean().clamp(min=1e-8)

        raw_d = raw_d / tgt_ref

        self.d[:self.m] = raw_d
        self.d[self.m:] = 0.0

        stats = {
            "d_source_mean": self.d[:self.m].mean().item(),
            "d_source_std": self.d[:self.m].std().item(),
            "d_source_min": self.d[:self.m].min().item(),
            "d_source_max": self.d[:self.m].max().item(),
            "d_target_ref": tgt_ref.item(),
        }
        print(
            "[BEA] dynamic d: mean={:.4f}, std={:.4f}, "
            "min={:.4f}, max={:.4f}, tgt_ref={:.4f}".format(
                stats["d_source_mean"], stats["d_source_std"],
                stats["d_source_min"], stats["d_source_max"],
                stats["d_target_ref"],
            )
        )
        return stats

    @torch.no_grad()
    def update_d_classifier(self, data_loader, obs_normalization_stats=None):
        """
        Recompute per-source discrepancy d_i via a domain classifier trained
        on obs-encoder embeddings.  d_i = P(source | x_i): source samples
        that look target-like get low d, clearly-source samples get high d.

        Source embeddings are subsampled to match the target count for
        balanced classifier training, then the fitted classifier is applied
        to all source samples.
        """
        from robomimic.utils.domain_classifier import compute_d_from_domain_classifier

        indices, embeddings = self._extract_obs_embeddings(
            data_loader, obs_normalization_stats,
        )

        src_emb = embeddings[:self.m]
        tgt_emb = embeddings[self.m:]

        d_per_sample, clf_info = compute_d_from_domain_classifier(src_emb, tgt_emb)

        self.d[:self.m] = torch.from_numpy(d_per_sample)
        self.d[self.m:] = 0.0

        stats = {
            "d_classifier_auc": clf_info["auc"],
            "d_source_mean": self.d[:self.m].mean().item(),
            "d_source_std": self.d[:self.m].std().item(),
            "d_source_min": self.d[:self.m].min().item(),
            "d_source_max": self.d[:self.m].max().item(),
        }
        print(
            "[BEA] domain classifier d: AUC={:.4f}, mean={:.4f}, std={:.4f}, "
            "min={:.4f}, max={:.4f}".format(
                clf_info["auc"],
                stats["d_source_mean"], stats["d_source_std"],
                stats["d_source_min"], stats["d_source_max"],
            )
        )
        return stats

    # Batch processing — pass through sample indices

    def process_batch_for_training(self, batch):
        """
        Extends parent to also pass through the "index" field, which is
        needed to look up q weights per sample.
        """
        input_batch = super().process_batch_for_training(batch)
        if "index" in batch:
            input_batch["index"] = batch["index"].long().to(self.device)
        return input_batch

    # Per-sample diffusion loss (no batch reduction)

    def _diffusion_forward(self, batch):
        """
        Run the diffusion forward pass and return per-sample MSE losses.

        Returns:
            per_sample_loss: (B,) tensor — MSE averaged over (Tp, ac_dim)
                but not over batch.
            info: dict with intermediate tensors for logging.
        """
        actions = batch["actions"]
        B = actions.shape[0]

        inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
        for k in self.obs_shapes:
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True
        )
        obs_cond = obs_features.flatten(start_dim=1)

        noise = torch.randn(actions.shape, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = self.nets["policy"]["noise_pred_net"](
            noisy_actions, timesteps, global_cond=obs_cond
        )

        # Per-sample MSE: average over (time, action_dim), keep batch dim
        # noise_pred and noise have shape (B, Tp, ac_dim)
        per_sample_loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2))

        return per_sample_loss, {"noise_pred": noise_pred, "noise": noise}

    # q-update step

    @torch.no_grad()
    def update_q(self, batch_indices, per_sample_losses, r_dp_value=0.0):
        """
        Projected subgradient step on q for the given batch indices.

        For each i in batch_indices:
            g_i = l_i + lambda_d * d_i
                  + gamma * r_dp * (1 if q_i == ||q||_inf else 0)
                  + lambda_1 * sign(q_i - p0_i) + 2 * lambda_2 * q_i
            q_i = q_i - eta_q * g_i
            q_i = clip(q_i, 0, q_max)

        If loss_ema_beta > 0, the EMA of losses is updated first and the
        smoothed loss is used in the subgradient instead of the raw loss.
        This reduces variance from random diffusion timestep/noise sampling.

        If target_q_min > 0, target sample weights are clamped from below
        to prevent them from collapsing to zero.

        The optional sum-constraint projection is NOT applied here — it is
        applied once after the full pass over the dataset (see
        project_q_simplex).

        Args:
            batch_indices: (B,) long tensor (CPU or GPU) — global dataset indices
            per_sample_losses: (B,) float tensor (CPU or GPU) — l(h(x_i), y_i)
            r_dp_value: scalar — R_DP(h) (weight-decay regularizer), precomputed.
        """
        cfg = self.algo_config.bea
        idx = batch_indices.cpu()
        losses = per_sample_losses.detach().cpu()

        # Optional: update and use loss EMA for variance reduction
        if self.loss_ema is not None:
            beta = cfg.loss_ema_beta
            self.loss_ema[idx] = beta * self.loss_ema[idx] + (1.0 - beta) * losses
            losses_for_grad = self.loss_ema[idx]
        else:
            losses_for_grad = losses

        q = self.q[idx]
        p0 = self.p0[idx]
        d = self.d[idx]

        # Subgradient of the full objective w.r.t. q_i
        # sign(0) = 0 which is a valid subgradient of |u| at u=0
        grad = (
            losses_for_grad
            + cfg.d.lambda_d * d
            + cfg.lambda_1 * torch.sign(q - p0)
            + 2.0 * cfg.lambda_2 * q
        )

        # Subgradient of gamma * R_DP * ||q||_inf w.r.t. q_i:
        # nonzero only for elements achieving the global max.
        gamma = self._get_gamma()
        if gamma > 0 and r_dp_value > 0:
            q_inf = self.q.max().item()
            is_max = (q - q_inf).abs() < 1e-8
            n_max = is_max.float().sum().clamp(min=1.0)
            grad = grad + gamma * r_dp_value * (is_max.float() / n_max)

        q = q - cfg.q.eta * grad
        q = q.clamp(0.0, cfg.q.max)

        # Enforce target weight floor if configured
        target_q_min = cfg.q.target_min
        if target_q_min > 0:
            target_mask = idx >= self.m
            q[target_mask] = q[target_mask].clamp(min=target_q_min)

        self.q[idx] = q

    def project_q_simplex(self):
        """
        Project q onto the constraint set
            {q : sum(q) = target_sum, lo_i <= q_i <= q_max}
        where target_sum = n + alpha * m, and lo_i = target_min for
        target samples (i >= m) and 0 for source samples (i < m).

        Uses bisection on the Lagrange multiplier nu:
            q_i^*(nu) = clip(q_i - nu, lo_i, q_max)
        Find nu such that sum_i q_i^*(nu) = target_sum.

        ONLY called if sum_constraint_enabled is True.
        """
        cfg = self.algo_config.bea
        if not cfg.q.sum_constraint_enabled:
            if cfg.q.target_min > 0 and self.n > 0:
                self.q[self.m:] = self.q[self.m:].clamp(min=cfg.q.target_min)
            return

        target_sum = float(self.n) + cfg.q.sum_constraint_alpha * float(self.m)
        q = self.q
        q_max = cfg.q.max

        q_lo = torch.zeros_like(q)
        if cfg.q.target_min > 0 and self.n > 0:
            q_lo[self.m:] = cfg.q.target_min
        q_hi = torch.full_like(q, q_max)

        # Feasibility check
        min_possible = q_lo.sum().item()
        max_possible = q_hi.sum().item()
        if target_sum < min_possible or target_sum > max_possible:
            print(
                f"[BEA] WARNING: target_sum={target_sum:.2f} infeasible "
                f"(range [{min_possible:.2f}, {max_possible:.2f}]). Clamping."
            )
            target_sum = max(min_possible, min(max_possible, target_sum))

        # Bisection: find nu such that sum(clip(q - nu, q_lo, q_hi)) = target_sum
        # sum is non-increasing in nu.
        lo_nu = float(q.min()) - q_max - 1.0
        hi_nu = float(q.max()) + 1.0

        for _ in range(64):
            mid = (lo_nu + hi_nu) / 2.0
            projected = torch.clamp(q - mid, min=q_lo, max=q_hi)
            current_sum = projected.sum().item()
            if current_sum > target_sum:
                lo_nu = mid
            else:
                hi_nu = mid
            if abs(current_sum - target_sum) < 1e-6:
                break

        nu = (lo_nu + hi_nu) / 2.0
        self.q = torch.clamp(q - nu, min=q_lo, max=q_hi)

    # Global q solver

    @torch.no_grad()
    def solve_q_global(self, all_losses, r_dp_value=0.0):
        """
        Global q solver

        Given all per-sample losses l_i (with h fixed), solve:

            min_{q}  sum_i q_i * c_i  +  gamma * R_DP * ||q||_inf
                     +  lambda_1 * ||q - p0||_1  +  lambda_2 * ||q||_2^2
            s.t.     q_lo_i <= q_i <= q_max       for all i
                     sum_i q_i = target_sum        (if sum_constraint_enabled)

        where c_i = l_i + lambda_d * d_i  and  gamma * R_DP is precomputed.

        The ||q||_inf term couples all elements.  We handle it by introducing
        t = ||q||_inf as an auxiliary variable and searching over t:

            F(t) = min_{q: q_i <= t}  [separable objective]  +  gamma * R_DP * t

        For each candidate t, the inner problem is separable with the same
        closed-form per element (but q_hi capped at min(q_max, t)).
        F(t) is convex, so we use ternary search over t in [q_lo_max, q_max].

        When gamma * R_DP == 0, this reduces to the original solver with no
        ||q||_inf term.

        Args:
            all_losses: (m+n,) tensor of per-sample losses, indexed 0..m+n-1.
            r_dp_value: scalar — R_DP(h) (weight-decay regularizer), precomputed.
        """
        cfg = self.algo_config.bea
        m, n = self.m, self.n
        total = m + n

        c = all_losses.clone()
        c[:m] += cfg.d.lambda_d * self.d[:m]

        # The paper assumes ℓ ∈ [0, 1].  Diffusion MSE losses are unbounded,
        # so normalize the cost vector to [0, 1] to preserve the intended
        # balance between the loss term and the λ₁/λ₂ regularizers.
        c_max = c.max().clamp(min=1e-8)
        if c_max > 1.0:
            c = c / c_max

        p0 = self.p0
        lam1 = cfg.lambda_1
        lam2 = cfg.lambda_2
        q_max_cfg = cfg.q.max

        q_lo = torch.zeros(total, dtype=torch.float32)
        if cfg.q.target_min > 0 and n > 0:
            q_lo[m:] = cfg.q.target_min

        gamma = self._get_gamma()
        gamma_r = gamma * r_dp_value

        def _solve_per_element(mu, q_hi):
            """Closed-form for  min (c_i-mu)*q_i + lam1*|q_i-p0_i| + lam2*q_i^2."""
            eff = c - mu
            residual = eff + 2.0 * lam2 * p0
            q_star = torch.where(
                residual < -lam1,
                (-eff - lam1) / (2.0 * lam2),
                torch.where(
                    residual > lam1,
                    (-eff + lam1) / (2.0 * lam2),
                    p0,
                ),
            )
            return torch.clamp(q_star, min=q_lo, max=q_hi)

        def _separable_obj(q_vec):
            """Evaluate the separable part: sum c_i*q_i + lam1*|q_i-p0_i| + lam2*q_i^2."""
            return (
                (c * q_vec).sum()
                + lam1 * (q_vec - p0).abs().sum()
                + lam2 * q_vec.pow(2).sum()
            ).item()

        def _solve_inner(t):
            """Solve the separable inner problem for a given t (upper bound on q_i).

            Returns (q_solution, separable_objective_value).
            """
            q_hi = torch.full((total,), min(q_max_cfg, t), dtype=torch.float32)
            q_hi = torch.max(q_hi, q_lo)

            if not cfg.q.sum_constraint_enabled:
                q_sol = _solve_per_element(0.0, q_hi)
                return q_sol, _separable_obj(q_sol)

            target_sum = float(n) + cfg.q.sum_constraint_alpha * float(m)

            mu_lo = float(c.min()) - lam1 - 2.0 * lam2 * q_max_cfg - 1.0
            mu_hi = float(c.max()) + lam1 + 1.0
            for _ in range(20):
                if _solve_per_element(mu_lo, q_hi).sum().item() >= target_sum:
                    mu_lo -= abs(mu_lo) + 10.0
                else:
                    break
            for _ in range(20):
                if _solve_per_element(mu_hi, q_hi).sum().item() <= target_sum:
                    mu_hi += abs(mu_hi) + 10.0
                else:
                    break

            for _ in range(100):
                mu_mid = (mu_lo + mu_hi) / 2.0
                q_mid = _solve_per_element(mu_mid, q_hi)
                s = q_mid.sum().item()
                if abs(s - target_sum) < 1e-6:
                    break
                if s < target_sum:
                    mu_lo = mu_mid
                else:
                    mu_hi = mu_mid

            q_sol = _solve_per_element((mu_lo + mu_hi) / 2.0, q_hi)
            return q_sol, _separable_obj(q_sol)

        print(
            f"[BEA] solve_q_global: c_norm range=[{c.min():.4f}, {c.max():.4f}], "
            f"lam1={lam1}, lam2={lam2}, p0_src={p0[:m].mean():.4f}, "
            f"p0_tgt={p0[m:].mean():.4f}, gamma_r={gamma_r:.4f}"
        )

        if gamma_r <= 0:
            q_hi = torch.full((total,), q_max_cfg, dtype=torch.float32)
            if not cfg.q.sum_constraint_enabled:
                self.q = _solve_per_element(0.0, q_hi)
            else:
                q_sol, _ = _solve_inner(q_max_cfg)
                self.q = q_sol
            return

        # With gamma_r > 0: search over t = ||q||_inf.
        # F(t) = inner_obj(t) + gamma_r * t is convex → ternary search.
        t_lo = q_lo.max().item()
        t_hi = q_max_cfg

        # When the sum constraint is active, t must be large enough that
        # sum_i min(q_max, t) >= target_sum, otherwise the inner problem
        # is infeasible.  The tightest simple bound is t >= target_sum / total.
        if cfg.q.sum_constraint_enabled:
            target_sum = float(n) + cfg.q.sum_constraint_alpha * float(m)
            t_lo = max(t_lo, target_sum / float(total))

        def _total_obj(t):
            _, sep_val = _solve_inner(t)
            return sep_val + gamma_r * t

        for _ in range(80):
            if t_hi - t_lo < 1e-7:
                break
            m1 = t_lo + (t_hi - t_lo) / 3.0
            m2 = t_hi - (t_hi - t_lo) / 3.0
            if _total_obj(m1) < _total_obj(m2):
                t_hi = m2
            else:
                t_lo = m1

        t_opt = (t_lo + t_hi) / 2.0
        q_sol, _ = _solve_inner(t_opt)
        self.q = q_sol

    # q-weighted training step

    def train_on_batch(self, batch, epoch, validate=False):
        """
        q-weighted training step.

        Computes per-sample diffusion loss, weights by q[index], averages
        over the batch, and does a gradient step on the model parameters.

        Returns an info dict with losses and q-statistics for logging.
        """
        assert self._bea_initialized, (
            "BEA weights not initialized. Call model.init_bea_weights(m, n) "
            "before training."
        )

        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = OrderedDict()
            assert validate or self.nets.training

            per_sample_loss, _ = self._diffusion_forward(batch)

            # Look up q weights for this batch
            indices = batch["index"].long()
            q_batch = self.q[indices.cpu()].to(self.device).detach()

            # Weighted mean loss: (1/B) * sum_i q_i * l_i
            weighted_loss = (q_batch * per_sample_loss).mean()
            # Also track unweighted loss for comparison
            unweighted_loss = per_sample_loss.mean()

            losses = {
                "l2_loss": unweighted_loss,
                "weighted_loss": weighted_loss,
                "q_mean": q_batch.mean(),
                "q_std": q_batch.std() if B > 1 else torch.tensor(0.0),
            }

            # Per-domain q stats
            src_mask = indices < self.m
            tgt_mask = ~src_mask
            if src_mask.any():
                losses["q_source_mean"] = q_batch[src_mask].mean()
            if tgt_mask.any():
                losses["q_target_mean"] = q_batch[tgt_mask].mean()

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

    # Compute losses for q-update pass (no gradient on model params)

    @torch.no_grad()
    def compute_losses_for_q_update(self, batch):
        """
        Forward pass to compute per-sample losses for the q-update phase.
        Called with model in eval mode and torch.no_grad().

        Args:
            batch: processed + postprocessed batch dict with "index" key.

        Returns:
            indices: (B,) long tensor — global dataset indices
            per_sample_losses: (B,) float tensor — per-sample MSE
        """
        per_sample_loss, _ = self._diffusion_forward(batch)
        indices = batch["index"].long()
        return indices, per_sample_loss

    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["l2_loss"].item()
        log["Weighted_Loss"] = info["losses"]["weighted_loss"].item()
        log["q_mean"] = info["losses"]["q_mean"].item()
        log["q_std"] = info["losses"]["q_std"].item()
        if "q_source_mean" in info["losses"]:
            log["q_source_mean"] = info["losses"]["q_source_mean"].item()
        if "q_target_mean" in info["losses"]:
            log["q_target_mean"] = info["losses"]["q_target_mean"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    # Serialization — save / load q weights alongside model

    def serialize(self):
        d = super().serialize()
        d["bea"] = {
            "q": self.q,
            "p0": self.p0,
            "d": self.d,
            "m": self.m,
            "n": self.n,
            "loss_ema": self.loss_ema,
        }
        return d

    def deserialize(self, model_dict, load_optimizers=False):
        super().deserialize(model_dict, load_optimizers=load_optimizers)
        if "bea" in model_dict:
            bea = model_dict["bea"]
            self.q = bea["q"]
            self.p0 = bea["p0"]
            self.d = bea["d"]
            self.m = bea["m"]
            self.n = bea["n"]
            self.loss_ema = bea.get("loss_ema", None)
            self._bea_initialized = True
