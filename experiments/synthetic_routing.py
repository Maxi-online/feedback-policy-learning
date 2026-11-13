"""Small CPU-only learning check. No conversations, LLMs or external services."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_learning import Config, PolicyLearner


def run(seed: int, updates: int, batch_size: int, test_size: int):
    random.seed(seed)
    torch.manual_seed(seed)
    config = Config(state_dim=12, action_dim=4, hidden_dim=64, learning_rate=0.003, epochs=4)
    learner = PolicyLearner(config)
    train_rng = torch.Generator().manual_seed(seed + 10000)
    test_rng = torch.Generator().manual_seed(seed + 20000)
    action_counts = [0] * config.action_dim
    for _ in range(updates):
        observations = torch.randn(batch_size, config.state_dim, generator=train_rng)
        labels = observations[:, :config.action_dim].argmax(dim=-1)
        steps = []
        for features, label in zip(observations.tolist(), labels.tolist()):
            action_counts[label] += 1
            decision = learner.decide(features, training=True)
            reward = float(decision.action_id == label)
            steps.append(learner.observe(decision, reward, [0.0] * config.state_dim, terminated=True))
        metrics = learner.update(steps)
    # This set is generated separately and is never used in policy/reward updates.
    test = torch.randn(test_size, config.state_dim, generator=test_rng)
    labels = test[:, :config.action_dim].argmax(dim=-1)
    with torch.no_grad():
        predicted = learner.actor(test).argmax(dim=-1)
    majority = max(range(config.action_dim), key=action_counts.__getitem__)
    random_actions = torch.randint(config.action_dim, (test_size,), generator=test_rng)
    return {
        "seed": seed, "training_samples": updates * batch_size, "held_out_samples": test_size,
        "mean_test_reward": {
            "categorical_ppo": float((predicted == labels).float().mean()),
            "uniform_random": float((random_actions == labels).float().mean()),
            "training_majority_action": float((labels == majority).float().mean()),
            "known_rule_oracle": 1.0,
        }, "last_update": metrics, "config": asdict(config),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 43])
    parser.add_argument('--updates', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-size', type=int, default=2048)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.test_size) < 2:
        parser.error('updates, batch size and test size must all be at least two')
    torch.set_num_threads(1)
    result = {
        "task": "synthetic independent one-step routing; target = argmax(first four features)",
        "scope": "Algorithm learning check only; no production or conversation-quality conclusion",
        "split": "Independent seeded train/test generators; no test-driven tuning in this run",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "torch_threads": 1},
        "runs": [run(seed, args.updates, args.batch_size, args.test_size) for seed in args.seeds],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({"task": result['task'], "runs": [{"seed": r['seed'], **r['mean_test_reward']} for r in result['runs']]}, indent=2))


if __name__ == '__main__':
    main()
