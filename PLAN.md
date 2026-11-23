# Measles school outbreak costs — plan and ABM feasibility

Based on `measles-dashboard` at commit `24f6b9e`.

The dashboard uses two models:
- the plot comes from `measles_single_population.DashboardExperiment` (20 reps)
- the numbers come from `measles_efficiency.compute_new_infections` (1000 reps, pure Python, τ-leap with dt = 0.25 days)

Both return new infections split into unvaccinated and breakthrough (vaccinated) cases, and that split is all the cost model needs.

---

## Part 1 — Outbreak cost + counterfactual vaccination scenarios (plan)

### 1.1 Cost model (new file `cost_model.py`)

Every cost is a function of the number of cases only, applied **per simulation** so we get a distribution and not just a point estimate.

Cases per simulation: `C = I0 + new_unvax + new_vax`. Make it a toggle whether the index case(s) count; I suggest yes, since they still cost money.

| Component | Quantity per simulation | Unit cost | Notes |
|---|---|---|---|
| Isolation (school days missed by cases) | `C × d_iso` | — (reported as days) | `d_iso` ≈ school days from symptom onset to 4 days after rash. Placeholder: 6 school days |
| Parent lost productivity | `C × p_caregiver × d_caregiver` | daily wage (BLS) | `p_caregiver` = share of cases where a parent misses work (age-dependent; elementary ≈ high) |
| Outpatient / ED visits | `C × v_op` | cost per visit | Could split into outpatient and ED |
| Hospitalizations | `Binomial(new_unvax, p_hosp_unvax) + Binomial(new_vax, p_hosp_vax)` | cost per hospitalization | Use separate (lower) risk for breakthrough cases, since the model already splits them. Binomial draws add realistic noise; plain expectations are fine too |
| *(optional)* Public-health response | `C × c_ph` | per-case contact tracing / response cost | In published outbreak cost studies this is often the **largest** component, so include it as a toggle (default off if you want "medical + productivity" only) |
| *(optional)* Deaths | `C × p_death` | report as an expected count, not monetized | |

All defaults go in one `COST_DEFAULTS` dict, each with a source comment. **Defaults still need to be sourced from the literature.** Candidates to check:
- CDC measles clinical / hospitalization data, including the 2025 outbreak hospitalization share
- societal-cost studies of US measles outbreaks, e.g. the 2019 Clark County, WA outbreak and the 2011 US outbreaks response-cost paper
- BLS for wages

Treat the numbers I used anywhere else as placeholders.

API:
```python
def outbreak_costs(new_unvax, new_vax, I0, cost_params, rng) -> dict[str, np.ndarray]  # per-sim arrays
def summarize_costs(per_sim: dict, threshold) -> dict  # expected value (all sims) + 95% range | outbreak
```

**What to display.** The dashboard's current headline numbers are conditional on outbreak size ≥ threshold. For costs I recommend two numbers:
- **Expected cost per introduction** (mean over all simulations, including those that fizzle out). This is the right quantity for comparing vaccination scenarios, because higher coverage lowers both the probability and the size of an outbreak.
- **95% range if an outbreak occurs** (conditional on ≥ threshold), to stay consistent with the existing "likely outbreak size" card.

### 1.2 Counterfactual vaccination scenarios

- `counterfactual_rates(v0, n=2) -> [v0 + k/(n+1)·(1−v0) for k in 1..n]`. This gives 85 → 90, 95 and 70 → 80, 90; 100% stays 100%. Round to 0.1 %.
- Show the two rates as **editable inputs** that default to the computed values and reset whenever the baseline rate changes.
- Run every scenario with the **same seed** (common random numbers), so the "averted" numbers reflect vaccination, not Monte Carlo noise.
- The existing input-validity check (I0 ≤ unvaccinated students) only needs to hold for the baseline. `compute_new_infections` already caps `S_vax` at `N − I0`, so counterfactuals with very high coverage still run.

### 1.3 Dash wiring (avoid re-simulating when only costs change)

1. A new `dcc.Store('scenario_results')` holds the per-simulation `new_unvax` / `new_vax` arrays for baseline + 2 counterfactuals: 3 × 2 × 1000 ints, which is tiny. It is filled by a callback on `dashboard_params` and the counterfactual-rate inputs. Option: have the existing `update_graph` callback also output the baseline arrays so the baseline isn't simulated twice.
2. A `dcc.Store('cost_params')` is filled from the cost selectors, the same pattern as `create_params_from_selectors`.
3. An `update_costs` callback on (`scenario_results`, `cost_params`) does arithmetic only. It is instant, so moving a cost slider never triggers a re-simulation.
4. UI (`app_dynamic_graphics.py`): a "Outbreak costs & impact of higher vaccination" section under the spaghetti plot with:
   - a table with columns *Current (85%)*, *90%*, *95%* and rows for chance of ≥ threshold, expected cases, hospitalizations, outpatient visits, school days missed (isolation), parent work days lost, and expected total cost, plus "averted vs current" values
   - a grouped bar chart of expected cost by component and scenario
   - cost parameters in a new "Cost Parameters ▾" accordion (`app_selectors.py`), and methods/sources text in `app_static_graphics.py`
5. Tests (`tests_model.py`):
   - cost is linear in cases, and 0 cases gives a cost of 0 (when the index case is excluded)
   - `counterfactual_rates(0.85) == [0.90, 0.95]` and `counterfactual_rates(1.0) == [1.0, 1.0]`
   - expected cases are monotone non-increasing across scenarios under common random numbers

**Runtime.** Three scenarios × 1000 reps with the current pure-Python SEIR take about **0.75 s** at N = 500 (0.41 + 0.25 + 0.09 s), which is acceptable. A Numba port of the same SEIR (`abm/seir_numba.py`) gives the same statistics in **~17 ms for all three scenarios** (10.6 / 5.1 / 1.5 ms). I recommend making that port, which needs no other changes. It requires adding `numba` to `requirements.txt` and warming the JIT cache at container start.

**Limitation to state on the dashboard.** In this model, isolation days are an *accounting* cost; they don't reduce transmission. Quarantine of exposed unvaccinated students can't be derived from case counts at all, and that is what Part 2 addresses.

---

## Part 2 — Agent-based model: does it match, and is it fast enough?

Code: `abm/measles_abm.py` (model) and `abm/benchmark.py` (comparison). To run: `cd abm && ../.venv/bin/python benchmark.py`.

### Model
- One agent per student, with S/E/I/R state, vaccination status and individual timers.
- Students sit in **classrooms within grades within the school** (default: 6 grades, classes of ~22).
- Transmission is group-based: the hazard is β × (w_class · infectious share in class + w_grade · share in grade + w_school · share in school). Each step costs O(N) and no edge list is stored. With weights (0, 0, 1) and exponential periods, it is the same stochastic process as the dashboard SEIR.
- Infectious periods can be exponential (`shape=1`, matching the dashboard) or gamma (more realistic, which matters for timing isolation).
- **Isolation**: a case is detected `detection_delay_days` after becoming infectious (≈ rash onset). Its transmission is then reduced by `isolation_effectiveness`.
- **Quarantine**: on detection, unvaccinated S/E students in the case's class, grade or school are excluded for 21 days, with reduced contacts. **Student-days in quarantine are counted**, so they can feed straight into the Part 1 cost model.
- Numba-compiled, with early stopping when no one is in E or I. Each replicate gets its own seed, so serial and parallel runs give **identical** results (checked).

### Does it reproduce the SEIR? (N = 500, R0 = 15, 1 index case)

| | P(≥10 new) 85% | mean new 85% | P 90% | P 95% |
|---|---|---|---|---|
| Dashboard SEIR, 10k reps | 0.563–0.568 | 36.1–36.3 | — | — |
| ABM well-mixed, 10k reps | 0.568–0.577 | 36.8–37.3 | — | — |
| Dashboard SEIR, 1k reps | 0.557 | 35.5 | 0.367 | 0.052 |
| ABM well-mixed, 1k reps | 0.554 | 35.9 | 0.341 | 0.040 |
| ABM class/grade/school (0.5/0.2/0.3), β ×1 | 0.476 | 24.3 | 0.222 | 0.014 |
| ABM structured, β ×1.22 (calibrated at 85%) | 0.558 | 34.7 | 0.321 | 0.028 |

- **Well-mixed ABM equals the SEIR** within Monte Carlo error.
- **Classroom structure at the same R0 makes outbreaks smaller**, because infections saturate locally within classrooms. One calibration factor (β × ~1.22) restores the 85% result. At higher coverage the structured model then gives somewhat smaller outbreaks than the SEIR, so the counterfactual benefit of vaccination would look a bit larger. The mixing weights are guesses for now; I'd calibrate them to a school contact survey.
- Switching to **gamma-distributed periods** (shape 4) at 85% coverage raises P(≥10) from 0.56 to **0.76**. Exponential periods make early die-out more likely. This matters for any isolation model, because isolation at rash onset only makes sense with a realistic infectious period.

### Interventions (illustrative: 85%, structured + calibrated, gamma periods, untuned effectiveness)

| Scenario | P(≥10 new) | mean new cases | mean quarantine student-days |
|---|---|---|---|
| None | 0.764 | 49.3 | 0 |
| Isolation at rash (4 days after onset of infectiousness, 90% effective) | 0.650 | 32.0 | 0 |
| + class quarantine of unvaccinated (21 days) | 0.388 | 10.7 | 272 |
| + school-wide quarantine of unvaccinated | 0.014 | 2.8 | 2,070 |

This is the trade-off the SEIR can't express: fewer cases versus many more days of school missed.

### Speed: 1000 reps, Apple M3 Pro (11 cores)

Single scenario at default N = 500 and 85% coverage:

| Model | Time |
|---|---|
| Dashboard SEIR (pure Python, current) | 0.41 s |
| SEIR, Numba port | 0.011 s |
| ABM well-mixed, serial | 1.27 s |
| ABM structured + isolation + class quarantine, serial | 0.73 s |
| same, parallel (11 threads) | **0.12 s** |

Scaling with school size (structured + isolation + class quarantine):

| N | SEIR (current) | ABM serial | ABM parallel |
|---|---|---|---|
| 250 | 0.30 s | 0.26 s | 0.05 s |
| 500 | 0.41 s | 0.74 s | 0.12 s |
| 1000 | 0.53 s | 2.0 s | 0.34 s |
| 2000 | 0.66 s | 5.5 s | 0.93 s |
| 4000 | 0.77 s | 14.4 s | 2.4 s |

The first call costs ~2 s of JIT compilation. After that it loads from Numba's disk cache in ~0.1 s, so the Docker image should warm the cache at build or start time.

### Verdict and next steps
- **An ABM is viable for the public tool.** At typical school sizes (≤ 1000), 1000 reps take 0.1–0.3 s in parallel or 0.7–2 s serial. That is about the same as, or faster than, the current pure-Python SEIR. Three vaccination scenarios at N = 500 take ~0.3 s in parallel.
- **Caveat about the server.** Parallel speed means a single request uses every core. Under gunicorn with several concurrent users, the effective speed is closer to the serial numbers. Mitigations:
  - cap threads per request with `numba.set_num_threads`
  - cache results by parameter tuple (inputs are discrete sliders, so the cache hit rate should be high)
  - run 500 reps for the ABM
- **Algorithmic speedup if large schools matter.** Today each step loops over all N students. Two changes would make the cost scale with the number of infected students rather than N × steps, and would probably bring the serial ABM close to the Numba SEIR:
  - keep a list of active (E/I) agents
  - draw infections per group with a Binomial on the group's susceptible count
- **Modelling to-dos before relying on intervention results:**
  - the index case shouldn't be detected on the same schedule as later cases, since detection usually happens only after secondary cases appear
  - decide whether quarantined students keep community contacts
  - calibrate the mixing weights
  - a household layer (siblings) could be added cheaply as one more group type

---

## Update (2026-11-19): faster kernels, parallel scaling, browser feasibility

**Part 1 is implemented** in `dashboard/` (a copy of `measles-dashboard` at `24f6b9e` with the cost extension; the original repo is unchanged):
- `cost_parameters.toml` holds every cost and scenario default, and the UI is generated from it
- `cost_model.py` is the cost logic; `tests_cost_model.py` tests it (10 passing)
- a "Cost Parameters ▾" dropdown and a cost/scenario table and chart are added to the dashboard

**Faster ABM kernels** (`abm/measles_abm.py`, `algorithm=` reference / active / group; validated with `abm/validate_fast.py`, timed with `abm/timing.py`). Times are for 1000 reps at 85% coverage, with structured mixing, isolation and class quarantine, best of 3, in ms:

| N | SEIR (Numba) | reference | active list | group draws | group, 11 threads | group, JS (Node, 1 thread) |
|---|---|---|---|---|---|---|
| 250 | 6 | 263 | 241 | 41 | 8 | 91 |
| 500 | 10 | 754 | 655 | 102 | 21 | 248 |
| 1000 | 15 | 2087 | 1799 | 283 | 53 | 702 |
| 2000 | 22 | 5496 | 5008 | 780 | 137 | 1747 |
| 4000 | 33 | 14556 | 13227 | 1839 | 330 | 4637 |

The three kernels give the same statistics within Monte Carlo error at 20k reps, and so does the JS port.

**Parallel scaling** (reference kernel, N = 2000): 5 threads give 4.5×, 8 give 5.4× and 11 give 6.0×. The M3 Pro has 5 performance + 6 efficiency cores, and the efficiency cores add only ~1.5× together. Dynamic scheduling barely helps (6.4×). The JIT compile is excluded from these timings because it is cached.

**Browser:** the JS port (now `dashboard/assets/measles_models.js`) runs 1000 reps at N = 500 in 0.25 s on one thread (all 3 vaccination scenarios in 0.43 s). It is ~2.4× slower than Numba. With Web-Worker-style threads, N = 2000 drops from 1.85 s to 0.57 s with 5 workers.

## Update (2026-11-20): test dashboard with switchable model engines

All new code now lives in this folder; `measles-dashboard` is back to its original state.

`dashboard/` is the dashboard plus the cost section plus a **Model Engine (testing)** panel. Same interface; the engine dropdown switches between:

| Engine | Where it runs |
|---|---|
| Compartmental · Python/NumPy (current production code) | server |
| Compartmental · Python/Numba | server |
| Compartmental · JavaScript | browser |
| ABM · Python/Numba, 1 thread / all cores | server |
| ABM · JavaScript, 1 thread / Web Workers | browser |

ABM options in the panel: well-mixed vs class/grade/school mixing (β × 1.22), exponential vs gamma periods, isolation, quarantine scope. The number of simulations per scenario can also be changed (100–10,000), and a re-run button repeats a run with the same inputs. The timing card shows each scenario's simulation time and the end-to-end time from an input change to the redrawn figure.

Run: `cd dashboard && ../.venv/bin/python app.py`, then open http://localhost:8050.

Code: `model_engines.py` (server engines), `app_engines.py` (panel layout and callbacks), `assets/measles_models.js` (JS models + Web Worker pool; also loaded by `abm/bench_js.mjs`), `assets/engine_callbacks.js` (Dash clientside callbacks). Every engine returns the same result structure, so the figure, statistics and costs are computed by the same Python code whatever the engine.

Measured in headless Chrome on the M3 Pro, N = 500, 1000 simulations × 3 scenarios (85/90/95%), second run:

| Engine | Well-mixed, no interventions | Class/grade/school, gamma, isolation + class quarantine |
|---|---|---|
| Compartmental, NumPy (production) | 824 ms | — |
| Compartmental, Numba | 23 ms | — |
| Compartmental, JS | 72 ms | — |
| ABM, Numba 1 thread | 294 ms | 192 ms |
| ABM, Numba 11 threads | 62 ms | 46 ms |
| ABM, JS 1 thread | 1,011 ms | 621 ms |
| ABM, JS 11 Web Workers | 224 ms | 136 ms |

End-to-end adds ~60–70 ms locally (callbacks + plotting), plus network latency when deployed. In Chrome, JS is ~3.4× slower than Numba (2.4× in the earlier Node benchmark). The JS compartmental model is 11× faster than the current production NumPy code.

Checks: the JS compartmental model matches NumPy at 20k–100k reps (e.g., 95%: P(≥10) 0.0445 / 0.0430 vs NumPy 0.0435). The JS and Numba ABMs match (P = 0.574 / 0.572; conditional outbreak-size percentiles 32/64/72 vs 31/64/72). At 1000 simulations the lower end of the "likely outbreak size" range is noisy for every engine (it sits in a sparse part of the distribution), so it varies by ±10 between engines and seeds.

## Update (2026-11-20, later): natural-history periods, quarantine costs, ranges, per-scenario reruns

- **Single-scenario reruns.** Scenarios are cached one by one, on the server (`model_engines._cached_scenario`) and in the browser (`engine_callbacks.js`). Editing one counterfactual rate only simulates that rate, and the timing card marks the others "reused". The re-run button recomputes everything. Each scenario uses the same seed whatever the other rates are, so cached and recomputed results are identical.
- **ABM "Measles natural history" periods** replace "gamma shape 4":
  - incubation (to prodrome, i.e. to infectiousness): gamma, truncated to 5–18 days, with the truncated mean equal to the dashboard incubation period (10.5)
  - infectious period: 8 days (truncated to 6–10)
  - rash at the mid-point of each case's infectious period; detection and isolation 1 day after rash
  - β = R0 / dashboard infectious period (5 days = 4 days before rash + 1), so R0 = 15 keeps the compartmental definition
  - gamma shapes 12 and 64: 98% of each untruncated distribution lies inside the bounds
  - without isolation, R = 24
  - school-structure β multiplier recalibrated: ×1.20 (×1.22 with exponential periods)
  - with exponential periods, isolation is implicit (it ends the 5-day infectious period), so the checkbox is disabled
  - all models now report the first and last detection day of each simulation; detection is the end of the infectious compartment for exponential periods
- **Effect of the periods at 85% coverage, 500 students:**

  | Model | P(≥10 new) | Mean new infections |
  |---|---|---|
  | Compartmental | 57% | 37 |
  | ABM natural history + isolation | 85% | 56 |
  | ABM natural history, no isolation | 97% | 73 |

  The less variable infectious period makes chance extinction after an introduction much rarer.
- **Quarantine costs.**
  - Isolation and quarantine are costed separately for students and parents.
  - ABM with a quarantine scope: simulated student-days are used.
  - Compartmental engines, or ABM with quarantine "not simulated": whole-school exclusion is assumed. All unvaccinated students without measles stay home from the first detected case until 21 days after the last one, as Broomfield CO, Oxford PA and Portland OR did in 2026. This is a cost assumption only; the compartmental model does not simulate quarantine's effect on transmission.
  - New parameters: quarantine length (21 days) and share of quarantined students with a parent missing work (0.5, placeholder).
- **Cost table:** expected values are followed by the 95% range over all simulations.
