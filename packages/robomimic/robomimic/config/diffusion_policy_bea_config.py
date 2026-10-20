"""
Config for BEA Diffusion Policy

Extends the standard DiffusionPolicyConfig
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class BEADiffusionPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_bea"

    def algo_config(self):
        super(BEADiffusionPolicyConfig, self).algo_config()

        # lambda_1: coefficient on L1 deviation from reference  ||q - p0||_1
        self.algo.bea.lambda_1 = 0.01
        # lambda_2: coefficient on L2 regularization  ||q||_2^2
        self.algo.bea.lambda_2 = 0.001

        # q — weight vector settings
        # "full"      — solve global convex opt (paper-faithful argmin)
        # "minibatch" — projected subgradient descent per mini-batch
        self.algo.bea.q.mode = "full"
        # update q every K epochs (1 = every epoch)
        self.algo.bea.q.update_freq = 1
        # upper bound on each weight  0 <= q_i <= q_max
        self.algo.bea.q.max = 5.0
        # lower bound on q for target samples (0.0 = no floor)
        self.algo.bea.q.target_min = 1.0
        # step size for projected subgradient (only used in minibatch mode)
        self.algo.bea.q.eta = 0.01
        # sum constraint: sum_i q_i = n + alpha * m
        self.algo.bea.q.sum_constraint_enabled = True
        # alpha: target_sum = n + alpha * m
        self.algo.bea.q.sum_constraint_alpha = 0.5

        # d — discrepancy penalty settings
        # "d_hat"      — constant value for all source samples (set once)
        # "knn"        — per-source k-NN distance in encoder feature space
        # "classifier" — domain classifier, d = 2*(AUC - 0.5)
        self.algo.bea.d.mode = "d_hat"
        # k for k-NN distance (only used when mode="knn")
        self.algo.bea.d.k = 3
        # constant discrepancy value (only used when mode="d_hat")
        self.algo.bea.d.value = 1.0
        # lambda_d: coefficient on discrepancy penalty  sum_i q_i * d_i
        self.algo.bea.d.lambda_d = 0.1

        # reference weights p_0
        # if True, p0_i = 1.0 for each source sample (ignores p0_source)
        self.algo.bea.p0_source_uniform = False
        self.algo.bea.p0_source = 0.0
        # if True, p0_i = 1.0 for each target sample (ignores p0_target)
        self.algo.bea.p0_target_uniform = True
        self.algo.bea.p0_target = 0.0

        # loss EMA for variance reduction (0.0 = disabled)
        self.algo.bea.loss_ema_beta = 0.0
