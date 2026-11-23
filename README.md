# Measles school outbreak simulator: costs and agent-based model

Test version of the [epiENGAGE measles outbreak simulator](https://epiengage-measles.tacc.utexas.edu/)
with:

- outbreak costs (isolation, quarantine, medical care) for the current and higher vaccination rates;
- an agent-based model (classes / grades / school, measles natural history, isolation and quarantine);
- a panel to switch between simulation engines (compartmental vs agent-based, Python vs JavaScript).

## Create the environment

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which also installs
Python if needed (Python 3.11 or later). From the repository root:

```bash
uv sync
```

This creates `.venv/` with the exact package versions in `uv.lock`.

## Run the app

```bash
cd dashboard
uv run python app.py
```

Then open <http://localhost:8050>. The app must be started from the `dashboard` folder
(it reads its data files from there). The first launch takes longer because Numba compiles
the simulation code; the compiled code is cached for later launches.

## Run the tests

```bash
cd dashboard
uv run pytest tests_cost_model.py tests_model_engines.py
```

## Where things are

| Path | Contents |
| --- | --- |
| `dashboard/app.py` | Dash app (main file) |
| `dashboard/cost_parameters.toml` | Default cost parameters, with sources |
| `dashboard/cost_model.py` | Cost calculations |
| `dashboard/model_engines.py`, `dashboard/app_engines.py` | Simulation engines and the engine panel |
| `dashboard/assets/` | JavaScript engines (run in the browser) |
| `abm/measles_abm.py` | Agent-based model (Numba) |
| `PLAN.md` | Design notes, validation and benchmark results |

`dashboard/README.md` is the README of the original dashboard (Docker deployment).
