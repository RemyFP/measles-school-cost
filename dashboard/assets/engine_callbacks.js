// Clientside (browser) callbacks for the model-engine testing panel.
// Registered in app_engines.py with ClientsideFunction("engines", <name>).
//
//   build_request : collects the inputs into one "sim_request" and starts the
//                   end-to-end clock (input change -> figure drawn).
//   run_browser   : runs the JavaScript engines (assets/measles_models.js) and
//                   returns the same structure as model_engines.run_server_engine.
//                   Scenarios are cached individually (as on the server), so
//                   changing one counterfactual rate only simulates that one.
//   end_to_end    : stops the clock once the outbreak figure has been redrawn.
const SCENARIO_CACHE = new Map();   // key -> scenario result (most recent last)
const SCENARIO_CACHE_SIZE = 64;
const toArr = (x) => (x === undefined || x === null ? null : Array.from(x));

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  engines: {
    // args: rerun clicks, params, valid, engine, structure, periods, isolation,
    //       quarantine, nReps, counterfactual rates..., config (State, last)
    build_request: function (...args) {
      const config = args.pop();
      const [rerun, params, valid, engine, structure, periods, isolation, quarantine, nReps,
             ...cfRates] = args;
      window.__measlesT0 = performance.now();
      const base = 100 * params.vax_prop[0];
      const rates = [Math.round(10 * base) / 10];
      cfRates.forEach((r, i) => {
        const dflt = Math.round(10 * (base + (i + 1) / (cfRates.length + 1) * (100 - base))) / 10;
        rates.push(r === null || r === undefined || r < 0 || r > 100 ? dflt : Number(r));
      });
      return {
        engine: engine, valid: valid, params: params, vax_rates_pct: rates,
        abm_settings: { structure: structure, periods: periods,
                        isolation: (isolation || []).includes("on"), quarantine: quarantine },
        n_reps: Number(nReps), n_curves: config.n_curves, seed: config.seed,
        is_browser: config.browser_engines.includes(engine), rerun: rerun || 0,
      };
    },

    run_browser: async function (req, config) {
      if (!req || !req.valid || !req.is_browser) return window.dash_clientside.no_update;
      const M = window.MeaslesModels;
      const p = req.params;
      const model = config.engine_model[req.engine];
      const useWorkers = req.engine === "abm_js_workers";
      const nWorkers = useWorkers ? M.maxWorkers() : 1;
      const seed = req.seed % 2 ** 31;

      // model parameters for one set of ABM settings (same mapping as model_engines.abm_params)
      const buildCommon = (s) => {
        const common = {
          R0: p.R0, incubation_period: p.incubation_period, infectious_period: p.infectious_period,
          relative_infectiousness_vaccinated: p.relative_infectiousness_vaccinated,
          vaccine_efficacy: p.vaccine_efficacy, population: p.population[0], I0: p.I0[0],
          sim_duration_days: p.sim_duration_days, time_step_days: p.time_step_days,
        };
        if (model === "abm") {
          const q = config.abm_quarantine[s.quarantine];
          Object.assign(common, config.abm_structures[s.structure], {
            beta_multiplier: s.structure === "school" ? config.abm_school_beta[s.periods] : 1.0,
            beta_days: p.infectious_period,
          }, config.abm_periods[s.periods], {
            isolation: s.isolation && s.periods !== "exponential",
            quarantine: q !== null, quarantine_scope: q === null ? 0 : q,
          });
          common.mixing_weights = Array.from(common.mixing_weights);
          if (common.incubation_min_days !== undefined) {
            common.incubation_min_days = Math.min(common.incubation_min_days, 0.5 * p.incubation_period);
            common.incubation_max_days = Math.max(common.incubation_max_days, 1.5 * p.incubation_period);
          }
        }
        return common;
      };

      // vaccination scenarios with the selected quarantine, then (ABM) the
      // current rate with each other quarantine option (as in run_server_engine)
      const sel = req.abm_settings.quarantine;
      const runs = req.vax_rates_pct.map((rate) => ({ rate: rate, q: model === "abm" ? sel : null, role: "rate" }));
      if (model === "abm") {
        Object.keys(config.abm_quarantine).filter((q) => q !== sel).forEach((q) =>
          runs.push({ rate: req.vax_rates_pct[0], q: q, role: "quarantine" }));
      }

      const tStart = performance.now();
      const scenarios = [], scenarioMs = [], reused = [];
      let curves = null, curveNewTotal = null, workersStarted = false;
      for (let i = 0; i < runs.length; i++) {
        const rate = runs[i].rate;
        const common = buildCommon(Object.assign({}, req.abm_settings, { quarantine: runs[i].q || sel }));
        const prm = Object.assign({}, common, { vax_prop: rate / 100 });
        // curves for every current-rate run, so switching quarantine reuses the cache
        const nCurves = i === 0 || runs[i].role === "quarantine" ? req.n_curves : 0;
        const key = JSON.stringify([req.engine, prm, req.n_reps, seed, nCurves, req.rerun]);
        let entry = SCENARIO_CACHE.get(key);
        reused.push(entry !== undefined);
        if (entry === undefined) {
          const t0 = performance.now();
          let r;
          if (useWorkers) {
            r = await M.runParallel(model, prm, req.n_reps, seed, nCurves, nWorkers);
            workersStarted = workersStarted || r.workersStarted;
          } else {
            r = (model === "abm" ? M.runABM : M.runSEIR)(prm, req.n_reps, seed, 0, nCurves);
          }
          const ms = Math.round(10 * (performance.now() - t0)) / 10;
          const sc = {
            vax_rate_pct: rate, new_unvax: toArr(r.new_unvax), new_vax: toArr(r.new_vax),
            first_detect_day: toArr(r.first_detect_day), last_detect_day: toArr(r.last_detect_day),
            // ABM: always "simulated" (zeros without quarantine -> no quarantine cost)
            quarantine_days: toArr(r.quarantine_days),
            n_quarantined: toArr(r.n_quarantined),
          };
          entry = { scenario: sc, ms: ms, curves: r.curves.map((c) => Array.from(c)) };
        } else {
          SCENARIO_CACHE.delete(key);   // re-insert below = most recently used
        }
        SCENARIO_CACHE.set(key, entry);
        while (SCENARIO_CACHE.size > SCENARIO_CACHE_SIZE) SCENARIO_CACHE.delete(SCENARIO_CACHE.keys().next().value);
        scenarioMs.push(entry.ms);
        scenarios.push(Object.assign({}, entry.scenario, { role: runs[i].role, quarantine: runs[i].q }));
        if (i === 0) {
          curves = entry.curves;
          curveNewTotal = entry.scenario.new_unvax.slice(0, nCurves)
            .map((u, k) => u + entry.scenario.new_vax[k]);
        }
      }
      return {
        engine: req.engine, valid: true, I0: p.I0[0], threshold: p.threshold_values[0],
        population: p.population[0], quarantine_scope: req.abm_settings.quarantine,
        scenarios: scenarios, curves: curves, curves_are_ma7: false,
        curve_new_total: curveNewTotal,
        timing: {
          total_ms: Math.round(10 * (performance.now() - tStart)) / 10, curves_ms: null,
          scenario_ms: scenarioMs, reused: reused, threads: nWorkers, where: "browser", n_reps: req.n_reps,
          workers_started: workersStarted,
        },
      };
    },

    end_to_end: function (figure, simResults) {
      if (!simResults || !simResults.valid || window.__measlesT0 === undefined) return "";
      // wait for the browser to paint the new figure before stopping the clock
      return new Promise((resolve) => requestAnimationFrame(() => setTimeout(() => {
        const ms = performance.now() - window.__measlesT0;
        resolve(`End-to-end (input change → figure drawn, includes network and plotting): ${ms.toFixed(0)} ms`);
      }, 0)));
    },
  },
});
