"""
Config for MSBEA (Multi-Source BEA) Diffusion Policy.

Extends BEADiffusionPolicyConfig with source-domain weight (w) settings.
"""

from robomimic.config.diffusion_policy_bea_config import BEADiffusionPolicyConfig


class MSBEADiffusionPolicyConfig(BEADiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_msbea"

    def algo_config(self):
        super(MSBEADiffusionPolicyConfig, self).algo_config()

        # Override BEA q.mode default — MSBEA only supports minibatch
        self.algo.bea.q.mode = "minibatch"

        # rho_1: L1 penalty on ||w - w0||_1
        self.algo.msbea.rho_1 = 0.01
        # rho_2: L2 penalty on ||w||_2^2
        self.algo.msbea.rho_2 = 0.001

        # w — source domain weight settings
        self.algo.msbea.w.eta = 0.01
        self.algo.msbea.w.min = 0.0
        self.algo.msbea.w.max = 1.0

        # step size for target q subgradient (may differ from source eta)
        self.algo.msbea.q.eta_T = 0.01
