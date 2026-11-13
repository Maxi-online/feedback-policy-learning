"""Opt-in SQLite storage of numeric trajectories, without conversation payloads."""
from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
from .core import Decision, Transition


class TrajectoryStore:
    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS rollouts (id INTEGER PRIMARY KEY, policy_version INTEGER NOT NULL, payload TEXT NOT NULL)")

    def append(self, steps: list[Transition]) -> int:
        if not steps or len({x.decision.policy_version for x in steps}) != 1:
            raise ValueError("A stored rollout must contain a single policy version")
        payload = json.dumps([asdict(step) for step in steps], allow_nan=False)
        with self.connection:
            cursor = self.connection.execute("INSERT INTO rollouts(policy_version,payload) VALUES (?,?)", (steps[0].decision.policy_version, payload))
        return int(cursor.lastrowid)

    def read(self, rollout_id: int) -> list[Transition]:
        row = self.connection.execute("SELECT payload FROM rollouts WHERE id=?", (rollout_id,)).fetchone()
        if row is None:
            raise KeyError(rollout_id)
        result = []
        for item in json.loads(row[0]):
            decision = item.pop("decision")
            decision["features"] = tuple(decision["features"])
            result.append(Transition(Decision(**decision), **item))
        return result

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
