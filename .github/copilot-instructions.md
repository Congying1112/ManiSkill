## Quick orientation for AI coding agents

This repo is ManiSkill (robotics simulation & training). The goal of this file is to give an AI code agent the minimal, high-value context to be immediately productive: the high-level architecture, where to find canonical examples, important patterns/conventions, and common developer workflows.

1) Big picture (what to read first)
  - `ManiSkill/README.md` — high level description, installation notes and links to docs.
  - Core package: `mani_skill/` contains the runtime code (agents, envs, sensors, utils).
  - Key directories:
    - `mani_skill/agents/` — robot implementations and controller helpers.
    - `mani_skill/envs/` — environment base classes and tasks. See `sapien_env.py` for the main Env lifecycle.
    - `mani_skill/sensors/` — camera & depth sensors and sensor config patterns.
    - `mani_skill/utils/` — asset management, building/URDF loaders, RNG and helpers.
    - `docs/source/user_guide/tutorials/` — curated examples (e.g., `custom_robots.md`).

2) Architectural highlights (how components interact)
  - Environments create a `ManiSkillScene` (see `mani_skill/envs/sapien_env.py`) and then load agents and actors into it.
  - Agents subclass `BaseAgent` (`mani_skill/agents/base_agent.py`): they supply `urdf_path`/`mjcf_path`, sensors (`_sensor_configs`), and controllers (`_controller_configs`).
  - Agent registration: agents may be registered with `@register_agent()` (`mani_skill/agents/registration.py`) — environments use `REGISTERED_AGENTS` in `_load_agent` to instantiate by id.
  - Sensors are described by config dataclasses (e.g., `CameraConfig`) and instantiated in `_setup_sensors`; observation modes are handled via `obs_mode` in `BaseEnv`.
  - GPU vs CPU simulation: `BaseEnv` picks sim backend automatically when `num_envs > 1` (GPU sim). `sim_freq` must be divisible by `control_freq` — code enforces this.

3) Project-specific conventions and patterns to follow
  - Register runtime artifacts (agents) with `@register_agent(...)` so they can be referenced by UID in env creation. See `mani_skill/agents/registration.py` and examples in `mani_skill/agents/robots/`.
  - Agent assets: agent classes can specify `asset_download_ids` in registration. Assets are stored under `MS_ASSET_DIR` or `~/.maniskill/data` by default (`mani_skill/__init__.py`). Use existing download helpers (search for `download_asset`) rather than inventing new download logic.
  - Controller configs return either a ControllerConfig or a dict of ControllerConfigs. Default controllers live in `mani_skill/agents/controllers/` and are referenced by name in `_controller_configs`.
  - Sensors use config-first approach: return camera/sensor config objects (e.g., `CameraConfig`) from `_sensor_configs` and let `BaseEnv._setup_sensors` instantiate them.
  - When modifying observation shapes or structure, call `BaseEnv.update_obs_space()` so the cached observation spaces are recomputed.

4) Developer workflows (how to build / run / test locally)
  - Install from PyPI for quick experiments: `pip install --upgrade mani_skill` (see `README.md`).
  - For local development from source: `pip install -e .[dev]` (setup.py defines `dev` extras). Tests expect assets to be available (see `tests/run.sh` which downloads assets in CI/docker).
  - Running tests: `pytest -n auto --forked tests` is used in the included `tests/run.sh` docker script; local quick checks can run a small subset, e.g. `pytest tests/test_envs.py -q`.
  - Quick local demos: examples in `mani_skill/examples/` (e.g., `demo_robot.py`, `demo_random_action.py`). The docs tutorial `custom_robots.md` shows a simple workflow to register and test a robot using `demo_robot`.

5) Important runtime invariants & gotchas for code changes
  - Simulation control vs simulation step: `sim_freq % control_freq == 0` is required (see `BaseEnv.__init__`). Changing control loops must preserve this invariant.
  - GPU sim initialization: building/loading assets on GPU requires careful ordering (see `_reconfigure` and `_setup_sensors` in `sapien_env.py`). Avoid changing sensor/scene setup order unless you understand GPU init semantics.
  - Determinism: the code manages two RNGs (main RNG and batched episode RNG). Use `_batched_episode_rng` for repeatable GPU/CPU parallel experiments.
  - Asset locations: code uses `MS_ASSET_DIR` env var to override the default `~/.maniskill` path; respect this when writing tests or temporary fixtures.

6) Where to find canonical examples to copy or extend
  - Robot template and registrations: `mani_skill/agents/robots/_template/template_robot.py` and many real robots under `mani_skill/agents/robots/`.
  - Task & env implementations: `mani_skill/envs/tasks/` and their corresponding docs under `docs/source/tasks/`.
  - Controller examples: `mani_skill/agents/controllers/` and usage examples inside agent implementations like `mani_skill/agents/robots/panda/`.

7) When modifying public APIs or behavior
  - Prefer adding small unit tests under `tests/` that exercise the new behavior and run them locally (or via `tests/run.sh` in docker for full matrix).
  - Update docs in `docs/` when the behavior or public API changes (many docs are generated from code; see `docs/generate_*_docs.py`).

If any important area above is unclear or you'd like the instructions to include more examples (e.g., common grep patterns, specific test commands, or a short dev checklist), tell me which section to expand and I'll iterate.
