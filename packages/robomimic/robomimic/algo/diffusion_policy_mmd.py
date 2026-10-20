"""
Maximum Mean Discrepancy (MMD) variant of Diffusion Policy.

Extends DiffusionPolicyUNet with an energy-distance MMD loss (via geomloss)
to align encoder feature distributions between source and target domains,
combined with standard diffusion BC loss on a dedicated BC portion.

Batch layout: [src_mmd(B_ot) | tgt_mmd(B_ot) | bc(remainder)]
Loss:  l2_diffusion + scale * MMD_energy

The energy distance is computed by geomloss.SamplesLoss("energy") on the
flattened encoder features from the source and target batch portions.
Unlike OT, this does not solve a transport plan — it directly penalises
the distributional gap with O(n^2) pairwise kernel evaluations.

Reference: geomloss — https://www.kernel-operations.io/geomloss/
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

from geomloss import SamplesLoss


@register_algo_factory_func("diffusion_policy_mmd")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the MMD DiffusionPolicy class to instantiate.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs
    """
    if algo_config.unet.enabled:
        return MMDDiffusionPolicyUNet, {}
    else:
        raise RuntimeError("Only UNet backbone is supported for MMD Diffusion Policy.")


class MMDDiffusionPolicyUNet(DiffusionPolicyUNet):
    """
    Diffusion Policy with MMD (energy distance) domain alignment.

    Inherits the full DiffusionPolicyUNet (network creation, inference,
    EMA, serialization) and overrides training-related methods to add
    an energy-distance alignment loss between source/target encoder features.
    """

    _debug_step = 0

    def _create_networks(self):
        super()._create_networks()
        self.mmd_loss_fn = SamplesLoss("energy")

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
        since the MMD portion may come from a source domain with different
        action semantics.

        Args:
            batch (dict): raw batch from data loader
            B_ot (int): number of source (= target) MMD alignment samples

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
        Training step with MMD alignment + diffusion BC.

        Batch layout: [src_mmd(B_ot) | tgt_mmd(B_ot) | bc(remainder)]

        Steps:
          1. Encode all observations through the shared encoder.
          2. Compute energy-distance MMD between source/target features.
          3. Compute standard diffusion noise-prediction loss on BC portion.
          4. Combine: loss = l2_loss + scale * mmd_loss.

        Args:
            batch (dict): processed batch from process_batch_for_training
            B_ot (int): number of source (= target) alignment samples
            ot_params (dict): alignment hyperparameters (uses "scale" key)
            epoch (int): current epoch
            validate (bool): if True, skip gradient updates

        Returns:
            info (dict): losses and diagnostics for logging
        """
        B_src = B_tgt = B_ot
        B = batch["actions"].shape[0]
        scale = ot_params.get("scale", 1.0)

        with TorchUtils.maybe_no_grad(no_grad=validate):
            # PolicyAlgo.train_on_batch initialises the info dict
            info = super(DiffusionPolicyUNet, self).train_on_batch(
                batch, epoch, validate=validate
            )

            obs_features = self._encode_obs(batch)

            # MMD alignment on encoder features
            mmd_src_feat = obs_features[:B_src].reshape(B_src, -1)
            mmd_tgt_feat = obs_features[B_src:B_src + B_tgt].reshape(B_tgt, -1)

            M_embed = torch.cdist(mmd_src_feat, mmd_tgt_feat) ** 2
            mmd_loss = self.mmd_loss_fn(mmd_src_feat, mmd_tgt_feat)

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
            loss = l2_loss + scale * mmd_loss

            MMDDiffusionPolicyUNet._debug_step += 1
            if MMDDiffusionPolicyUNet._debug_step % 50 == 1:
                print(
                    f"[MMD step={MMDDiffusionPolicyUNet._debug_step}] "
                    f"B_ot={B_ot} B_bc={B_bc} | "
                    f"M_embed: mean={M_embed.mean().item():.4f} | "
                    f"mmd_loss={mmd_loss.item():.4f} "
                    f"scale*mmd={scale * mmd_loss.item():.6f} | "
                    f"l2_loss={l2_loss.item():.4f} total={loss.item():.4f}"
                )

            losses = {
                "l2_loss": l2_loss,
                "mmd_loss": mmd_loss,
                "total_loss": loss,
            }
            info["losses"] = TensorUtils.detach(losses)
            info["M_embed"] = TensorUtils.detach(M_embed)

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

    # Logging

    def log_info(self, info):
        """Summarize training info for tensorboard logging."""
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["bc_loss"] = info["losses"]["l2_loss"].item()
        log["mmd_loss"] = info["losses"]["mmd_loss"].item()
        log["total_loss"] = info["losses"]["total_loss"].item()

        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "enc_grad_norm" in info:
            log["enc_grad_norm"] = info["enc_grad_norm"]
        if "unet_grad_norm" in info:
            log["unet_grad_norm"] = info["unet_grad_norm"]

        if "M_embed" in info:
            log["M_embed"] = (
                info["M_embed"].cpu().numpy()
                if isinstance(info["M_embed"], torch.Tensor)
                else info["M_embed"]
            )

        return log
