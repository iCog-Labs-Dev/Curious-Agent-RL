"""Unified experiment runner for Curious-Agent-RL.

By default, this entry point trains the tabular agent followed by the DQN
agent. Either experiment can also be run independently.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIGS = {
    "tabular": PROJECT_ROOT / "configs" / "tabular.yaml",
    "dqn": PROJECT_ROOT / "configs" / "dqn.yaml",
    "vanilla-dqn": PROJECT_ROOT / "configs" / "vanilla_dqn.yaml",
    "icm": PROJECT_ROOT / "configs" / "icm.yaml",
}

logger = logging.getLogger(__name__)


def run_tabular(config: dict, render: bool) -> None:
    """Load and run the tabular trainer."""
    from scripts.train_tabular import train_tabular

    train_tabular(config, render=render)


def run_dqn(config: dict, render: bool) -> None:
    """Load and run the DQN trainer.

    Uses the vectorized trainer when multi-env is enabled.
    """
    num_envs = config.get("training", {}).get("num_envs", 1)
    if num_envs > 1:
        from scripts.train_dqn_vec import train_dqn_vec
        train_dqn_vec(config, render=render)
    else:
        from scripts.train_dqn import train_dqn
        train_dqn(config, render=render)


def run_vanilla_dqn(config: dict, render: bool) -> None:
    """Load and run the external-reward-only DQN trainer."""
    from scripts.train_vanilla_dqn import train_vanilla_dqn

    train_vanilla_dqn(config, render=render)


def run_icm(config: dict, render: bool) -> None:
    """Load and run the ICM (Pathak) DQN trainer.

    Uses the vectorized trainer when multi-env is enabled.
    """
    num_envs = config.get("training", {}).get("num_envs", 1)
    if num_envs > 1:
        from scripts.train_dqn_icm_vec import train_dqn_icm_vec
        train_dqn_icm_vec(config, render=render)
    else:
        from scripts.train_dqn_icm import train_dqn_icm
        train_dqn_icm(config, render=render)


TRAINERS: dict[str, Callable[[dict, bool], None]] = {
    "tabular": run_tabular,
    "dqn": run_dqn,
    "vanilla-dqn": run_vanilla_dqn,
    "icm": run_icm,
}


def load_config(config_path: Path) -> dict:
    """Load and validate an experiment configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    return config


def set_seed(seed: int, include_torch: bool = False) -> None:
    """Seed the random number generators used by an experiment."""
    random.seed(seed)
    np.random.seed(seed)

    if include_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def prepare_config(
    config: dict,
    agent_name: str,
    episodes: int | None = None,
    max_steps: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Return a copy of a config with command-line overrides applied."""
    prepared = copy.deepcopy(config)
    training = prepared.setdefault("training", {})

    if episodes is not None:
        training["num_episodes"] = episodes
    if max_steps is not None:
        training["max_steps_per_episode"] = max_steps

    if output_dir is not None:
        agent_output = output_dir / agent_name
        prepared["paths"] = {
            "checkpoints": str(agent_output / "checkpoints"),
            "logs": str(agent_output / "logs"),
            "results": str(agent_output / "results"),
        }

    return prepared


def selected_agents(agent_option: str) -> list[str]:
    """Expand the requested pipeline into an execution order."""
    if agent_option == "all":
        return ["tabular", "vanilla-dqn", "dqn", "icm"]
    if agent_option == "both":
        return ["tabular", "dqn"]
    if agent_option == "dqn-pair":
        return ["vanilla-dqn", "dqn"]
    if agent_option == "curious-pair":
        return ["dqn", "icm"]
    return [agent_option]


def _resolve_log_files(
    agent_names: list[str],
    output_dir: Path | None,
) -> dict[str, Path]:
    """Map agent names to their training log files after a pipeline run."""
    log_files: dict[str, Path] = {}
    for name in agent_names:
        if output_dir is not None:
            candidate = output_dir / name / "logs" / "training.log"
        elif name == "vanilla-dqn":
            candidate = Path("logs") / "vanilla_dqn" / "training.log"
        else:
            candidate = Path(f"training_{name}.log")

        if candidate.is_file():
            log_files[name] = candidate.resolve()
    return log_files


AGENT_LABELS: dict[str, str] = {
    "tabular": "Tabular Q-Learning",
    "vanilla-dqn": "Vanilla DQN",
    "dqn": "Curious DQN (Schmidhuber)",
    "icm": "ICM DQN (Pathak)",
}


def _generate_comparison_plot(
    agent_names: list[str],
    output_dir: Path | None,
    labels: list[str] | None = None,
) -> None:
    """Generate a coupled comparison plot from multi-agent pipeline logs."""
    try:
        from scripts.visualize import plot_comparison
    except ImportError:
        logger.warning("Could not import plot_comparison; skipping comparison plot.")
        return

    log_files = _resolve_log_files(agent_names, output_dir)
    if len(log_files) < 2:
        logger.info(
            "Found %d log file(s); need at least 2 for a comparison plot.",
            len(log_files),
        )
        return

    paths = [str(p) for p in log_files.values()]
    resolved_labels = labels or [
        AGENT_LABELS.get(name, name) for name in log_files
    ]
    logger.info("Generating comparison plot from: %s", ", ".join(log_files))
    try:
        plot_comparison(paths, labels=resolved_labels)
    except Exception:
        logger.warning("Comparison plot generation failed.", exc_info=True)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the experiments selected by parsed command-line arguments."""
    config_paths = {
        "tabular": Path(args.tabular_config).expanduser().resolve(),
        "dqn": Path(args.dqn_config).expanduser().resolve(),
        "vanilla-dqn": Path(args.vanilla_dqn_config).expanduser().resolve(),
        "icm": Path(args.icm_config).expanduser().resolve(),
    }
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else None
    )

    experiments: list[tuple[str, dict, int]] = []
    for agent_name in selected_agents(args.agent):
        config = load_config(config_paths[agent_name])
        config = prepare_config(
            config,
            agent_name=agent_name,
            episodes=args.episodes,
            max_steps=args.max_steps,
            output_dir=output_dir,
        )
        seed = (
            args.seed
            if args.seed is not None
            else int(config.get("training", {}).get("seed", 42))
        )
        config.setdefault("training", {})["seed"] = seed
        experiments.append((agent_name, config, seed))

    logger.info("Experiment pipeline: %s", " -> ".join(selected_agents(args.agent)))
    for agent_name, config, seed in experiments:
        training = config.get("training", {})
        paths = config.get("paths", {})
        logger.info(
            "%s: episodes=%s, max_steps=%s, seed=%s, checkpoints=%s",
            agent_name,
            training.get("num_episodes"),
            training.get("max_steps_per_episode"),
            seed,
            paths.get("checkpoints"),
        )

    if args.dry_run:
        logger.info("Dry run complete; no experiments were started.")
        return

    pipeline_start = time.monotonic()
    for agent_name, config, seed in experiments:
        logger.info("Starting %s experiment", agent_name)
        set_seed(seed, include_torch=agent_name in {"dqn", "vanilla-dqn", "icm"})
        experiment_start = time.monotonic()
        TRAINERS[agent_name](config, render=args.render)
        logger.info(
            "Finished %s experiment in %.2f seconds",
            agent_name,
            time.monotonic() - experiment_start,
        )

    logger.info(
        "Full experiment pipeline completed in %.2f seconds",
        time.monotonic() - pipeline_start,
    )

    if len(experiments) > 1 and not args.no_plot:
        _generate_comparison_plot(
            [name for name, _, _ in experiments],
            output_dir,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the Curious-Agent-RL experiment pipeline."
    )
    parser.add_argument(
        "--agent",
        choices=("tabular", "dqn", "vanilla-dqn", "icm", "dqn-pair", "curious-pair", "both", "all"),
        default="both",
        help=(
            "Experiment to run. 'dqn-pair' runs the vanilla baseline followed "
            "by curiosity-coupled DQN. 'curious-pair' runs the Schmidhuber DQN "
            "followed by the Pathak ICM DQN for a direct curiosity comparison. "
            "'all' runs all four: tabular, vanilla DQN, curious DQN, and ICM DQN."
        ),
    )
    parser.add_argument(
        "--tabular-config",
        default=str(DEFAULT_CONFIGS["tabular"]),
        help="Path to the tabular YAML configuration.",
    )
    parser.add_argument(
        "--dqn-config",
        default=str(DEFAULT_CONFIGS["dqn"]),
        help="Path to the DQN YAML configuration.",
    )
    parser.add_argument(
        "--vanilla-dqn-config",
        default=str(DEFAULT_CONFIGS["vanilla-dqn"]),
        help="Path to the vanilla DQN YAML configuration.",
    )
    parser.add_argument(
        "--icm-config",
        default=str(DEFAULT_CONFIGS["icm"]),
        help="Path to the ICM DQN YAML configuration.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Override the number of episodes for every selected experiment.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override the maximum steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the random seed; otherwise use each config's seed.",
    )
    parser.add_argument(
        "--output-dir",
        help="Store each agent's checkpoints, logs, and results below this directory.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment with Pygame during training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved experiment plan without training.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip automatic comparison plot generation after multi-agent runs.",
    )
    return parser


def positive_int(value: int | None, option: str) -> None:
    """Raise a parser-friendly error for non-positive training overrides."""
    if value is not None and value <= 0:
        raise ValueError(f"{option} must be greater than zero")


def main() -> None:
    """Parse arguments and run the selected experiment pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    try:
        positive_int(args.episodes, "--episodes")
        positive_int(args.max_steps, "--max-steps")
        run_pipeline(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
