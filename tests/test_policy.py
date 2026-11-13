from dataclasses import replace
from math import isfinite
from pathlib import Path
import tempfile
import unittest
import torch
from torch.distributions import Categorical
from policy_learning import Config, Decision, PolicyLearner, Transition, generalized_advantage
from policy_learning.storage import TrajectoryStore

torch.set_num_threads(1)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.learner = PolicyLearner(Config(state_dim=4, action_dim=3, hidden_dim=16, epochs=2))

    def rollout(self, count=16, training=True):
        result = []
        for i in range(count):
            features = [i / count, 0.2, 0.3, 0.4]
            d = self.learner.decide(features, training=training)
            result.append(self.learner.observe(d, float(d.action_id == i % 3), features, terminated=True))
        return result

    def test_recorded_log_probability_belongs_to_selected_action(self):
        decision = self.learner.decide([0.1, 0.2, 0.3, 0.4], training=True)
        expected = Categorical(logits=self.learner.actor(torch.tensor(decision.features))).log_prob(torch.tensor(decision.action_id))
        self.assertAlmostEqual(decision.behavior_log_probability, expected.item(), places=6)

    def test_gae_terminal_and_time_limit_have_different_bootstrap(self):
        d = Decision((0.0,), 0, -0.5, 2.0, 0, True)
        terminal = Transition(d, 3.0, 10.0, True, False)
        truncated = Transition(d, 3.0, 10.0, False, True)
        advantages, returns = generalized_advantage([truncated, terminal], 0.9, 0.95)
        self.assertEqual(advantages, [10.0, 1.0])
        self.assertEqual(returns, [12.0, 3.0])

    def test_update_changes_parameters_and_rejects_reuse(self):
        steps = self.rollout()
        before = [p.detach().clone() for p in self.learner.actor.parameters()]
        metrics = self.learner.update(steps)
        self.assertTrue(all(isfinite(x) for x in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, self.learner.actor.parameters())))
        with self.assertRaises(ValueError):
            self.learner.update(steps)

    def test_deterministic_evaluation_cannot_enter_training(self):
        with self.assertRaises(ValueError):
            self.learner.update(self.rollout(training=False))

    def test_invalid_features_and_rewards_fail_before_training(self):
        with self.assertRaises(ValueError):
            self.learner.decide([float('nan')] * 4)
        with self.assertRaises(ValueError):
            self.learner.decide([1.0])
        decision = self.learner.decide([0.0] * 4, training=True)
        with self.assertRaises(ValueError):
            self.learner.observe(decision, float('inf'), [0.0] * 4)

    def test_old_behavior_probability_is_used_in_ratio(self):
        steps = self.rollout()
        shifted = [replace(s, decision=replace(s.decision, behavior_log_probability=s.decision.behavior_log_probability - 1)) for s in steps]
        metrics = self.learner.update(shifted)
        self.assertGreater(metrics['approx_kl'], 0.4)
        self.assertGreater(metrics['clip_fraction'], 0.9)

    def test_checkpoint_roundtrip_preserves_predictions_and_version(self):
        self.learner.update(self.rollout())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            self.learner.save_checkpoint(path)
            restored = PolicyLearner.load_checkpoint(path)
            self.assertEqual(self.learner.decide([0.1] * 4), restored.decide([0.1] * 4))

    def test_numeric_store_roundtrip(self):
        steps = self.rollout()
        with TrajectoryStore(':memory:') as store:
            row = store.append(steps)
            self.assertEqual(steps, store.read(row))
            with self.assertRaises(KeyError):
                store.read(row + 1)

    def test_reward_training_is_explicit_and_does_not_change_policy(self):
        before = self.learner.decide([0.0] * 4)
        loss = self.learner.fit_reward_model([[0.0] * 4, [1.0] * 4], [0, 1], [0.1, 0.9])
        self.assertTrue(isfinite(loss))
        self.assertEqual(before, self.learner.decide([0.0] * 4))
        with self.assertRaises(ValueError):
            self.learner.fit_reward_model([[0.0] * 4], [0], [2.0])


if __name__ == '__main__':
    unittest.main()
