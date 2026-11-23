// node bench_js.mjs  -- single-thread timing + statistics, then worker_threads
// scaling (worker_threads ~ Web Workers in a browser).
import { createRequire } from "node:module";
// The JS models live in the test dashboard (classic script, also a CommonJS module)
const { runABM } = createRequire(import.meta.url)("../dashboard/assets/measles_models.js");
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import os from "node:os";

const PRM = { mixing_weights: [0.5, 0.2, 0.3], beta_multiplier: 1.22, incubation_shape: 4, infectious_shape: 4,
              isolation: true, quarantine: true, vax_prop: 0.85,
              rash_fraction: 0, detection_delay_days: 4, isolation_effectiveness: 0.9 };

if (!isMainThread) {
  const { prm, n, seed, start } = workerData;
  const r = runABM(prm, n, seed, start);
  parentPort.postMessage({ u: r.new_unvax, v: r.new_vax });
} else {
  const stats = (r) => {
    let big = 0, sum = 0, q = 0;
    for (let i = 0; i < r.new_unvax.length; i++) { const t = r.new_unvax[i] + r.new_vax[i]; sum += t; if (t >= 10) big++; q += r.quarantine_days[i]; }
    const n = r.new_unvax.length;
    return `P(>=10)=${(big / n).toFixed(3)} mean new=${(sum / n).toFixed(2)} quarantine days=${(q / n).toFixed(1)}`;
  };
  const best = (f, k = 3) => { let b = 1e9; for (let i = 0; i < k; i++) { const t = performance.now(); f(); b = Math.min(b, performance.now() - t); } return b; };

  console.log("Statistics, 20000 reps (compare with validate_fast.py):");
  console.log("  well-mixed 85%:                ", stats(runABM({ vax_prop: 0.85 }, 20000, 1)));
  console.log("  structured 90%:                ", stats(runABM({ ...PRM, isolation: false, quarantine: false, vax_prop: 0.90 }, 20000, 1)));
  console.log("  structured 85% iso+class quar.:", stats(runABM(PRM, 20000, 1)));

  console.log("\nSingle thread, 1000 reps, structured + isolation + class quarantine (best of 3):");
  for (const N of [250, 500, 1000, 2000, 4000]) {
    runABM({ ...PRM, population: N }, 50, 7); // warm-up JIT
    console.log(`  N=${String(N).padStart(4)}: ${best(() => runABM({ ...PRM, population: N }, 1000, 1)).toFixed(1)} ms`);
  }
  console.log("\n3 scenarios (85/90/95%), N=500, 1000 reps each:",
    best(() => { for (const v of [0.85, 0.9, 0.95]) runABM({ ...PRM, vax_prop: v }, 1000, 1); }).toFixed(1), "ms");

  // worker_threads scaling (includes worker startup, like a page spawning Web Workers)
  const runWorkers = (nw, prm, nReps) => new Promise((resolve) => {
    const per = Math.ceil(nReps / nw); let done = 0;
    for (let w = 0; w < nw; w++) {
      const wk = new Worker(new URL(import.meta.url), { workerData: { prm, n: Math.min(per, nReps - w * per), seed: 1, start: w * per } });
      wk.on("message", () => { wk.terminate(); if (++done === nw) resolve(); });
    }
  });
  console.log(`\nWorkers (fresh workers each run, incl. startup), N=2000, 1000 reps; cores=${os.cpus().length}`);
  for (const nw of [1, 2, 5, 11]) {
    let b = 1e9;
    for (let i = 0; i < 3; i++) { const t = performance.now(); await runWorkers(nw, { ...PRM, population: 2000 }, 1000); b = Math.min(b, performance.now() - t); }
    console.log(`  ${String(nw).padStart(2)} workers: ${b.toFixed(1)} ms`);
  }
}
