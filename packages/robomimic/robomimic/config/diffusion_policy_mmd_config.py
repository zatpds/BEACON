"""
Config for MMD Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig

class MMDDiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy_mmd"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 0.00

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # UNet parameters
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256, 512, 1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'

        ## Paired dataset params (shared with OT data pipeline)
        self.train.ot.pair_info_path = None
        self.train.ot.dataset = None
        self.train.ot.dataset_masks = None
        self.train.ot.batch_size = None

        ## DTW pairing params
        self.algo.ot.sharpness = 0.0
        self.algo.ot.cutoff = None
        self.algo.ot.window_size = 200
        self.algo.ot.no_window = True

        ## MMD loss scaling
        self.algo.ot.scale = 1.0

        ## Unused OT params kept for interface compatibility with run_epoch_for_ot_policy
        self.algo.ot.emb_scale = None
        self.algo.ot.cost_scale = None
        self.algo.ot.reg = None
        self.algo.ot.tau1 = None
        self.algo.ot.tau2 = None
        self.algo.ot.heuristic = False
        self.algo.ot.label = None
