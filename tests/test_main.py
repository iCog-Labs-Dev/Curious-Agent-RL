"""Tests for the unified experiment runner."""

from argparse import Namespace
from pathlib import Path

import main


def make_args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "agent": "both",
        "tabular_config": str(main.DEFAULT_CONFIGS["tabular"]),
        "dqn_config": str(main.DEFAULT_CONFIGS["dqn"]),
        "vanilla_dqn_config": str(main.DEFAULT_CONFIGS["vanilla-dqn"]),
        "icm_config": str(main.DEFAULT_CONFIGS["icm"]),
        "episodes": 2,
        "max_steps": 3,
        "seed": 7,
        "output_dir": str(tmp_path),
        "render": False,
        "dry_run": False,
        "no_plot": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_prepare_config_does_not_mutate_source(tmp_path):
    source = {
        "training": {"num_episodes": 100, "max_steps_per_episode": 20},
        "paths": {"checkpoints": "original"},
    }

    prepared = main.prepare_config(
        source,
        agent_name="tabular",
        episodes=2,
        max_steps=3,
        output_dir=tmp_path,
    )

    assert source["training"]["num_episodes"] == 100
    assert prepared["training"]["num_episodes"] == 2
    assert prepared["training"]["max_steps_per_episode"] == 3
    assert prepared["paths"]["checkpoints"] == str(
        tmp_path / "tabular" / "checkpoints"
    )


def test_pipeline_runs_both_agents_in_order(monkeypatch, tmp_path):
    calls = []

    def record(agent_name):
        def runner(config, render):
            calls.append(
                (
                    agent_name,
                    config["training"]["num_episodes"],
                    config["training"]["max_steps_per_episode"],
                    render,
                )
            )

        return runner

    monkeypatch.setitem(main.TRAINERS, "tabular", record("tabular"))
    monkeypatch.setitem(main.TRAINERS, "dqn", record("dqn"))
    main.run_pipeline(make_args(tmp_path))

    assert calls == [
        ("tabular", 2, 3, False),
        ("dqn", 2, 3, False),
    ]


def test_dry_run_does_not_start_training(monkeypatch, tmp_path):
    def fail_if_called(config, render):
        raise AssertionError("trainer should not run during a dry run")

    monkeypatch.setitem(main.TRAINERS, "tabular", fail_if_called)
    monkeypatch.setitem(main.TRAINERS, "dqn", fail_if_called)

    main.run_pipeline(make_args(tmp_path, dry_run=True))


def test_all_pipeline_runs_all_four_agents(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setitem(
        main.TRAINERS,
        "tabular",
        lambda config, render: calls.append("tabular"),
    )
    monkeypatch.setitem(
        main.TRAINERS,
        "vanilla-dqn",
        lambda config, render: calls.append("vanilla-dqn"),
    )
    monkeypatch.setitem(
        main.TRAINERS,
        "dqn",
        lambda config, render: calls.append("dqn"),
    )
    monkeypatch.setitem(
        main.TRAINERS,
        "icm",
        lambda config, render: calls.append("icm"),
    )

    main.run_pipeline(make_args(tmp_path, agent="all"))

    assert calls == ["tabular", "vanilla-dqn", "dqn", "icm"]


def test_dqn_pair_runs_vanilla_before_curiosity(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setitem(
        main.TRAINERS,
        "vanilla-dqn",
        lambda config, render: calls.append("vanilla-dqn"),
    )
    monkeypatch.setitem(
        main.TRAINERS,
        "dqn",
        lambda config, render: calls.append("dqn"),
    )

    main.run_pipeline(make_args(tmp_path, agent="dqn-pair"))

    assert calls == ["vanilla-dqn", "dqn"]


def test_seed_override_is_written_to_prepared_config(monkeypatch, tmp_path):
    configured_seeds = []
    monkeypatch.setitem(
        main.TRAINERS,
        "vanilla-dqn",
        lambda config, render: configured_seeds.append(config["training"]["seed"]),
    )

    main.run_pipeline(
        make_args(tmp_path, agent="vanilla-dqn", seed=123)
    )

    assert configured_seeds == [123]
