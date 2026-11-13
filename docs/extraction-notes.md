# Extraction and correctness notes

The source study was part of a larger private assistant application. This repository publishes an isolated, anonymized learning component with an explicit inspection-only proprietary notice. No customer runtime, commercial service identity, database, learned weights or conversations are included.

## Preserved concepts

- Two-hidden-layer actor and critic networks with configurable widths.
- A separate state/action reward-estimation MLP.
- Categorical routing, clipped policy updates, gradient clipping and explicit checkpoints.
- A trajectory persistence boundary, now numeric-only and opt-in.

## Corrected behavior

1. **Action identity:** the old action vector mixed continuous metadata and route one-hot values. Taking `argmax` across that vector was not a valid categorical action ID. `Decision.action_id` now stores the sampled ID directly. There is one configured categorical action space, without a ten-output-to-eight-route modulo mapping.
2. **Behavior policy:** old probabilities were recomputed immediately before training, rather than captured when the action was sampled. Every decision now stores its original log probability and policy version.
3. **On-policy sampling:** an epsilon/greedy mixture was previously treated as if it were a categorical sample. Training now samples categorically; greedy evaluation data is rejected by PPO.
4. **Returns:** the old update regressed immediate rewards without using terminal flags or next-state values. Ordered GAE now distinguishes terminal transitions from time-limit truncations.
5. **Replay:** policy versions reject stale rollout reuse. Persisted trajectories are available for inspection; persistence does not make them valid on-policy training data after an update.
6. **Reward meaning:** the actor's own confidence is no longer an automatic quality reward. Explicit reward-model ratings are separate from policy rewards and are range checked.
7. **Import behavior:** the old module created a global system, opened SQLite and attempted an emotion-model download at import. The extracted module has no such singleton or external model dependency.
8. **Data boundary:** the old experience schema stored user identifiers and message text. The extracted store accepts numeric transitions only.
9. **Checkpoint loading:** the loader explicitly requests tensor/state-dictionary loading using `weights_only=True`.

## Claims intentionally excluded

The original module's broad overview mentioned Decision Transformers, offline RL and meta-learning. Those labels do not establish completed algorithms or experiments. This repository documents only the implemented categorical on-policy learner and optional supervised reward estimator. Automatic emotion analysis is not relabeled as human preference data.

The original author-side files are retained locally. They are not copied into this repository as an unredacted appendix. The public history starts with the isolated source and contains no legacy application history.
