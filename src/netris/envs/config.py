"""Environment configuration."""

from dataclasses import dataclass, field


@dataclass
class NetrisEnvConfig:
    """Knobs that do not change the observation/action spaces."""

    include_next_piece: bool = False
    watchdog_steps: int = 1000
    collect_metrics: bool = True

    line_clear_rewards: dict = field(
        default_factory=lambda: {1: 5.0, 2: 15.0, 3: 40.0}
    )
    tetris_reward: float = 100.0
    hole_penalty: float = 0.5
    aggregate_height_penalty: float = 0.005
    bumpiness_penalty: float = 0.01
    game_over_penalty: float = -10.0
    survival_bonus: float = 1.0

    def to_dict(self):
        return {
            "include_next_piece": self.include_next_piece,
            "watchdog_steps": self.watchdog_steps,
            "collect_metrics": self.collect_metrics,
            "line_clear_rewards": dict(self.line_clear_rewards),
            "tetris_reward": self.tetris_reward,
            "hole_penalty": self.hole_penalty,
            "aggregate_height_penalty": self.aggregate_height_penalty,
            "bumpiness_penalty": self.bumpiness_penalty,
            "game_over_penalty": self.game_over_penalty,
            "survival_bonus": self.survival_bonus,
        }
