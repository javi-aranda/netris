import json
from collections import Counter
from datetime import UTC, datetime

import numpy as np


class MetricsCollector:
    """In-memory per-episode metrics.

    Persisting metrics to disk is the responsibility of the main-process
    callbacks: environments may run in subprocesses (SubprocVecEnv) and must
    not race on shared files.
    """

    def __init__(self):
        self._reset_episode()

    def _reset_episode(self):
        self.episode_actions = Counter()
        self.episode_rotations = 0
        self.episode_placement_columns = []
        self.episode_placement_heights = []
        self.episode_reward_breakdowns = []
        self.episode_line_clears = []
        self.episode_max_height = 0
        self.episode_bumpiness_values = []
        self.episode_holes_values = []

    def register_action(self, action):
        self.episode_actions[int(action)] += 1

    def register_rotation(self):
        self.episode_rotations += 1

    def register_piece_placed(self, column, height):
        self.episode_placement_columns.append(column)
        self.episode_placement_heights.append(height)

    def register_reward_breakdown(self, *, reward_lines=0.0, reward_holes=0.0,
                                   reward_height=0.0, reward_bumpiness=0.0,
                                   reward_clean=0.0, reward_total=0.0):
        self.episode_reward_breakdowns.append({
            "lines": reward_lines,
            "holes": reward_holes,
            "height": reward_height,
            "bumpiness": reward_bumpiness,
            "clean": reward_clean,
            "total": reward_total,
        })

    def register_line_clear(self, lines_delta):
        self.episode_line_clears.append(lines_delta)

    def update_max_height(self, height):
        self.episode_max_height = max(self.episode_max_height, height)

    def record_bumpiness(self, bumpiness):
        self.episode_bumpiness_values.append(bumpiness)

    def record_holes(self, holes):
        self.episode_holes_values.append(holes)

    def episode_summary(self, episode, steps, pieces, total_lines, total_reward):
        total_lines_from_breakdown = sum(
            d.get("lines", 0) for d in self.episode_reward_breakdowns
        )
        total_holes = sum(
            d.get("holes", 0) for d in self.episode_reward_breakdowns
        )
        total_height = sum(
            d.get("height", 0) for d in self.episode_reward_breakdowns
        )
        total_bumpiness = sum(
            d.get("bumpiness", 0) for d in self.episode_reward_breakdowns
        )

        avg_placement_col = (
            np.mean(self.episode_placement_columns)
            if self.episode_placement_columns else 0.0
        )
        avg_placement_height = (
            np.mean(self.episode_placement_heights)
            if self.episode_placement_heights else 0.0
        )
        avg_bumpiness = (
            np.mean(self.episode_bumpiness_values)
            if self.episode_bumpiness_values else 0.0
        )
        avg_holes = (
            np.mean(self.episode_holes_values)
            if self.episode_holes_values else 0.0
        )

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "episode": episode,
            "steps": steps,
            "pieces": pieces,
            "lines_cleared": total_lines,
            "max_height": self.episode_max_height,
            "avg_bumpiness": round(float(avg_bumpiness), 3),
            "avg_holes": round(float(avg_holes), 3),
            "reward_total": round(float(total_reward), 4),
            "reward_lines": round(total_lines_from_breakdown, 4),
            "reward_holes": round(total_holes, 4),
            "reward_height": round(total_height, 4),
            "reward_bumpiness": round(total_bumpiness, 4),
            "rotations": self.episode_rotations,
            "avg_placement_col": round(float(avg_placement_col), 3),
            "avg_placement_height": round(float(avg_placement_height), 3),
            "action_distribution": json.dumps(dict(self.episode_actions)),
            "line_clear_distribution": json.dumps(dict(Counter(self.episode_line_clears))),
        }

    def reset_episode(self):
        self._reset_episode()

    def get_placement_heatmap(self):
        heatmap = np.zeros((18, 10), dtype=np.int32)
        for col, height in zip(self.episode_placement_columns, self.episode_placement_heights):
            row = 18 - 1 - height
            row = max(0, min(17, row))
            col = max(0, min(9, col))
            heatmap[row, col] += 1
        return heatmap


class NullMetricsCollector:
    """No-op collector used when metrics collection is disabled."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def episode_summary(self, **kwargs):
        return {}
