# Learning and evaluation protocol

## On-policy objective

For a sampled action `a_t`, the decision record stores `log pi_old(a_t | s_t)` at collection time. A PPO update forms `r_t = exp(log pi_new - log pi_old)` and minimizes the negative mean of `min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)`. The complete objective adds the configured MSE value loss and subtracts the entropy bonus.

The TD residual is `reward + gamma * next_value - value`. The next-state value is zero for a true terminal state. GAE accumulates discounted residuals within an episode; both termination and truncation prevent a later episode's advantage from entering the preceding one. A truncation still allows the value bootstrap for its final observation. Returns are `advantage + recorded_value`. Advantage normalization uses population standard deviation.

The implementation processes a complete ordered rollout per epoch. It does not implement recurrent policies, asynchronous actors, vector-environment interleaving, importance-weighted offline replay, distributed learning, value clipping or automatic KL early stopping. Policy updates and environment collection must be coordinated by the caller.

## Synthetic routing study

Each observation is an independently sampled twelve-dimensional standard-normal vector. Four actions are available. The correct action is the `argmax` of dimensions zero through three; the remaining eight dimensions are nuisance features. This is a contextual bandit represented as one-step terminal episodes, so it tests categorical learning and actor/critic optimization, not long-horizon planning.

Default training uses seeds 7, 19 and 43, 32 updates per seed, 128 examples per update, four epochs per update, a 64-unit hidden width and learning rate 0.003. Each seed is evaluated on 2,048 independently generated examples. A different generator seed is used for testing, and no test sample is used in training or baseline fitting.

All policies see the same test examples. The uniform policy samples actions independently; the fixed-action policy uses the most frequent training label; the oracle applies the known generating rule. The reported mean reward equals routing accuracy for this binary-reward task. Raw per-seed results are stored, without pooling the runs into an unsupported production claim.

Hyperparameters are fixed in the script. This small study is not a search for a best published benchmark score, and the test distribution does not demonstrate robustness to distribution shift. There is no claim of statistical significance against a research baseline from three seeds.

## Application study still required

A real routing evaluation needs an authorized dataset, a documented action taxonomy and a measurable outcome independent of the current actor. Splits should isolate conversations and, where relevant, time periods. A static router, a simple supervised router and the existing application policy should be compared under the same observations and action constraints. Reward components should be ablated rather than assuming emotion-score changes measure satisfaction.

Any analysis of previously logged decisions needs the actual behavior propensities and an appropriate off-policy estimator; this repository's PPO API deliberately rejects stale logged data. A reward model needs its own held-out labels, error analysis and calibration assessment before its output becomes a training reward. Deployment also requires latency measurements, a fallback policy, drift monitoring and a controlled online study.

## Reproducibility boundary

The experiment seeds Python and PyTorch, uses one CPU thread and records the installed Python/PyTorch/platform versions. Exact floating-point agreement across platforms and library builds is not promised. The public repository contains no pretrained checkpoint; author-side tests and synthetic experiments generate their own state. Unit tests verify numerical and lifecycle contracts separately from the synthetic learning result.
