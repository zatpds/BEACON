"""
Unbalanced Optimal Transport (UOT) variant of Diffusion Policy.

Extends DiffusionPolicyUNet with a Sinkhorn-based UOT coupling loss between
source and target encoder features, combined with standard diffusion BC loss
on a dedicated BC portion of each batch.

Batch layout: [src_ot(B_ot) | tgt_ot(B_ot) | bc(remainder)]
Loss:  l2_diffusion + scale * OT_loss
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

import ot


@register_algo_factory_func("diffusion_policy_ot")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the OT DiffusionPolicy class to instantiate.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs
    """
    if algo_config.unet.enabled:
        return OTDiffusionPolicyUNet, {}
    else:
        raise RuntimeError("Only UNet backbone is supported for OT Diffusion Policy.")


class OTDiffusionPolicyUNet(DiffusionPolicyUNet):
    """
    Diffusion Policy with Unbalanced Optimal Transport domain alignment.

    Inherits the full DiffusionPolicyUNet
    """

    _debug_step = 0

    # Observation encoding helper

    def _encode_obs(self, batch):
        """
        Encode observations through the shared visual encoder.

        Inserts a time dimension when missing (happens when obs_horizon=1
        and the dataset omits the singleton time axis).

        Returns:
            obs_features: [B, T, D] tensor of encoder features.
        """
        inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k]) - 1:
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True
        )
        assert obs_features.ndim == 3  # [B, T, D]
        return obs_features

    # Batch processing

    def process_batch_for_training(self, batch, B_ot):
        """
        Processes input batch, truncating to observation/prediction horizons.

        Only validates action range on the BC portion (indices >= 2*B_ot),
        since the OT portion may come from a source domain with different
        action semantics.

        Args:
            batch (dict): raw batch from data loader
            B_ot (int): number of source (= target) OT alignment samples

        Returns:
            input_batch (dict): processed batch ready for training
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        if not self.action_check_done:
            actions = input_batch["actions"][2 * B_ot:]
            in_range = (-1 <= actions) & (actions <= 1)
            if not torch.all(in_range).item():
                raise ValueError(
                    '"actions" must be in [-1,1] for Diffusion Policy. '
                    "Check if hdf5_normalize_action is enabled."
                )
            self.action_check_done = True

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    # Training step

    def train_on_batch(self, batch, B_ot, ot_params, epoch, validate=False):
        """
        Training step with UOT alignment + diffusion BC.

        Batch layout: [src_ot(B_ot) | tgt_ot(B_ot) | bc(remainder)]

        Steps:
          1. Encode all observations through the shared encoder.
          2. Build cost matrix M = emb_scale*M_embed + cost_scale*M_label.
          3. Solve UOT coupling via Sinkhorn-Knopp with marginal relaxation.
          4. Compute standard diffusion noise-prediction loss on BC portion.
          5. Combine: loss = l2_loss + scale * ot_loss.

        Args:
            batch (dict): processed batch from process_batch_for_training
            B_ot (int): number of source (= target) alignment samples
            ot_params (dict): UOT hyperparameters — keys:
                scale, emb_scale, cost_scale, reg, tau1, tau2,
                label ("eef_pose" | "action"), heuristic (bool)
            epoch (int): current epoch
            validate (bool): if True, skip gradient updates

        Returns:
            info (dict): losses and diagnostics for logging
        """
        B_src = B_tgt = B_ot
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            # PolicyAlgo.train_on_batch initialises the info dict
            info = super(DiffusionPolicyUNet, self).train_on_batch(
                batch, epoch, validate=validate
            )

            obs_features = self._encode_obs(batch)

            # OT alignment on encoder features
            src_feat = obs_features[:B_src].reshape(B_src, -1)
            tgt_feat = obs_features[B_src:B_src + B_tgt].reshape(B_tgt, -1)

            M_embed = torch.cdist(src_feat, tgt_feat) ** 2

            # Label-based cost component: anchors transport on task semantics
            if ot_params["label"] == "eef_pose":
                src_label = batch["obs"]["robot0_eef_pos"][:B_src].reshape(B_src, -1).detach()
                tgt_label = batch["obs"]["robot0_eef_pos"][B_src:B_src + B_tgt].reshape(B_tgt, -1).detach()
            else:
                src_label = batch["actions"][:B_src, :1].reshape(B_src, -1).detach()
                tgt_label = batch["actions"][B_src:B_src + B_tgt, :1].reshape(B_tgt, -1).detach()

            M_label = torch.cdist(src_label, tgt_label) ** 2

            M = (
                ot_params["emb_scale"] * M_embed
                + ot_params["cost_scale"] * M_label.to(M_embed.device)
            )

            # Optional diagonal-biased initialization for the coupling
            if ot_params["heuristic"]:
                src_n, tgt_n = src_feat.shape[0], tgt_feat.shape[0]
                c = np.zeros((src_n, tgt_n))
                c[:src_n, :src_n] = np.eye(src_n) / src_n
            else:
                c = None

            # Solve unbalanced OT: tau1/tau2 control marginal relaxation
            a, b = ot.unif(src_feat.shape[0]), ot.unif(tgt_feat.shape[0])
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a, b, M.detach().cpu().numpy(),
                ot_params["reg"],
                c=c,
                reg_m=(ot_params["tau1"], ot_params["tau2"]),
            )

            pi_sum = pi.sum(dtype=np.float32)
            pi_diag = np.trace(pi, dtype=np.float32)

            pi = torch.from_numpy(pi).float().to(M_embed.device)
            ot_loss = torch.sum(pi * M)

            OTDiffusionPolicyUNet._debug_step += 1
            if OTDiffusionPolicyUNet._debug_step % 50 == 1:
                M_np = M.detach().cpu().numpy()
                scaled_ot = ot_params["scale"] * ot_loss.item()
                print(
                    f"[OT step={OTDiffusionPolicyUNet._debug_step}] "
                    f"B_ot={B_ot} B_bc={B - 2 * B_ot} | "
                    f"M: min={np.min(M_np):.6f} mean={np.mean(M_np):.4f} "
                    f"max={np.max(M_np):.4f} | "
                    f"M_embed: mean={M_embed.mean().item():.4f} | "
                    f"M_label: mean={M_label.mean().item():.6f} | "
                    f"pi: sum={pi_sum:.6f} diag={pi_diag:.6f} "
                    f"max={pi.max().item():.6f} | "
                    f"ot_loss={ot_loss.item():.4f} scale*ot={scaled_ot:.6f}"
                )

            # BC diffusion portion
            bc_feat = obs_features[B_src + B_tgt:]
            bc_actions = batch["actions"][B_src + B_tgt:]
            B_bc = bc_actions.shape[0]

            obs_cond = bc_feat.flatten(start_dim=1)
            noise = torch.randn(bc_actions.shape, device=self.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B_bc,), device=self.device,
            ).long()

            noisy_actions = self.noise_scheduler.add_noise(bc_actions, noise, timesteps)
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond
            )

            l2_loss = F.mse_loss(noise_pred, noise)
            loss = l2_loss + ot_params["scale"] * ot_loss

            losses = {
                "l2_loss": l2_loss,
                "ot_loss": ot_loss,
                "total_loss": loss,
            }
            info["losses"] = TensorUtils.detach(losses)
            info["pi_sum"] = pi_sum
            info["pi_diag"] = pi_diag
            info["M_embed"] = TensorUtils.detach(M_embed)
            info["M_label"] = TensorUtils.detach(M_label)
            info["pi"] = pi

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                enc_grad = sum(
                    p.grad.data.norm(2).pow(2).item()
                    for p in self.nets["policy"]["obs_encoder"].parameters()
                    if p.grad is not None
                ) ** 0.5
                unet_grad = sum(
                    p.grad.data.norm(2).pow(2).item()
                    for p in self.nets["policy"]["noise_pred_net"].parameters()
                    if p.grad is not None
                ) ** 0.5
                info["enc_grad_norm"] = enc_grad
                info["unet_grad_norm"] = unet_grad

                if self.ema is not None:
                    self.ema.step(self.nets)

                info["policy_grad_norms"] = policy_grad_norms

        return info

    def log_info(self, info):
        """Summarize training info for tensorboard logging."""
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["bc_loss"] = info["losses"]["l2_loss"].item()
        log["ot_loss"] = info["losses"]["ot_loss"].item()
        log["total_loss"] = info["losses"]["total_loss"].item()

        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "enc_grad_norm" in info:
            log["enc_grad_norm"] = info["enc_grad_norm"]
        if "unet_grad_norm" in info:
            log["unet_grad_norm"] = info["unet_grad_norm"]

        for k in ("pi_sum", "pi_diag", "pi", "M_embed", "M_label"):
            if k in info:
                log[k] = info[k].cpu().numpy() if isinstance(info[k], torch.Tensor) else info[k]

        return log
