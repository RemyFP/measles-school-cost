// Measles outbreak models in JavaScript, to test running simulations in the
// user's browser instead of on the server.
//   runABM  : port of the "group" kernel in abm/measles_abm.py
//             (_simulate_fast_py, group_draws=True)
//   runSEIR : port of the dashboard's compartmental model
//             (measles_efficiency.compute_new_infections, tau-leap)
// No dependencies. The same file is used as
//   - a classic <script> in the page (Dash loads everything in assets/):
//     defines window.MeaslesModels, including a Web Worker pool;
//   - the Web Worker script (the pool starts workers from this file's URL);
//   - a CommonJS module in Node (abm/bench_js.mjs, validation scripts).
(function (root) {
"use strict";

const DEFAULTS = {
  R0: 15.0, incubation_period: 10.5, infectious_period: 5.0,
  relative_infectiousness_vaccinated: 0.05, vaccine_efficacy: 0.997,
  population: 500, I0: 1, vax_prop: 0.85, sim_duration_days: 250, time_step_days: 0.25,
  n_grades: 6, class_size: 22, mixing_weights: [0, 0, 1], beta_multiplier: 1.0,
  // shape 1 = exponential (dashboard SEIR); otherwise gamma truncated to [min, max]
  incubation_shape: 1.0, incubation_min_days: 0, incubation_max_days: Infinity,
  infectious_shape: 1.0, infectious_min_days: 0, infectious_max_days: Infinity,
  beta_days: null,              // beta = R0 / beta_days (null -> infectious_period)
  // detection at onset + rash_fraction * infectious period + delay (see measles_abm.py)
  rash_fraction: 1.0, detection_delay_days: 0.0,
  isolation: false, isolation_effectiveness: 1.0,
  quarantine: false, quarantine_scope: 0, quarantine_days: 21.0, quarantine_effectiveness: 0.9,
};

// ---------- RNG: sfc32 (fast, good quality, seedable) ----------
function makeRng(seed) {
  let a = 0x9e3779b9, b = 0x243f6a88, c = 0xb7e15162, d = seed >>> 0;
  const next = () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 15; i++) next();
  const random = () => { let u; do { u = next(); } while (u === 0); return u; }; // (0,1)
  const randint = (n) => Math.floor(next() * n);
  // number of trials up to and including first success (>=1)
  const geometric = (p) => (p >= 1 ? 1 : 1 + Math.floor(Math.log(random()) / Math.log1p(-p)));
  // exact binomial by summing geometric waiting times; cost O(k+1)
  const binomial = (n, p) => {
    if (p <= 0 || n === 0) return 0;
    if (p >= 1) return n;
    const lq = Math.log1p(-p);
    let k = 0, pos = 0;
    for (;;) {
      pos += 1 + Math.floor(Math.log(random()) / lq);
      if (pos > n) return k;
      k++;
    }
  };
  const normal = () => Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * next());
  const gamma = (shape, scale) => { // Marsaglia-Tsang, shape >= 1
    const dd = shape - 1 / 3, cc = 1 / Math.sqrt(9 * dd);
    for (;;) {
      let x, v;
      do { x = normal(); v = 1 + cc * x; } while (v <= 0);
      v = v * v * v;
      const u = random();
      if (Math.log(u) < 0.5 * x * x + dd - dd * v + dd * Math.log(v)) return dd * v * scale;
    }
  };
  // Poisson as in numpy: multiplication method for lam < 10, PTRS otherwise
  const poisson = (lam) => {
    if (lam <= 0) return 0;
    if (lam < 10) {
      const enlam = Math.exp(-lam);
      let x = 0, prod = 1.0;
      for (;;) { prod *= next(); if (prod > enlam) x++; else return x; }
    }
    const slam = Math.sqrt(lam), loglam = Math.log(lam);
    const b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
    const invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
      const U = next() - 0.5, V = next(), us = 0.5 - Math.abs(U);
      const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
      if (us >= 0.07 && V <= vr) return k;
      if (k < 0 || (us < 0.013 && V > us)) continue;
      if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <=
          -lam + k * loglam - logGamma(k + 1)) return k;
    }
  };
  return { random, randint, geometric, binomial, gamma, poisson };
}

// log Gamma(x) for x >= 1 (numpy's random_loggam)
const LG_COEF = [8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04,
  -5.952380952380952e-04, 8.417508417508418e-04, -1.917526917526918e-03,
  6.410256410256410e-03, -2.955065359477124e-02, 1.796443723688307e-01, -1.39243221690590e+00];
function logGamma(x) {
  if (x === 1 || x === 2) return 0;
  const n = x < 7 ? Math.floor(7 - x) : 0;
  let x0 = x + n;
  const x2 = 1 / (x0 * x0);
  let gl0 = LG_COEF[9];
  for (let k = 8; k >= 0; k--) gl0 = gl0 * x2 + LG_COEF[k];
  let gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * Math.log(x0) - x0;
  for (let k = 1; k <= n; k++) { gl -= Math.log(x0 - 1); x0 -= 1; }
  return gl;
}

// ---------- Compartmental model (port of measles_efficiency) ----------
// Each rep is seeded with seed + rep (same convention as runABM), so reps can
// be split across workers. curves[r][day] = E + I at the start of each day.
function runSEIR(params, nReps, seed = 12345, repStart = 0, nCurveReps = 0) {
  const p = { ...DEFAULTS, ...params };
  const N = p.population | 0, I0 = p.I0 | 0, dt = p.time_step_days;
  const nSteps = 1 + Math.floor(p.sim_duration_days / dt), stepsPerDay = Math.round(1 / dt);
  const nDays = Math.floor(p.sim_duration_days) + 1;
  const sVax0 = Math.min(Math.floor(N * p.vax_prop), N - I0);
  const beta = p.R0 / p.infectious_period, rv = p.relative_infectiousness_vaccinated;
  const ve = p.vaccine_efficacy, eOut = dt / p.incubation_period, iOut = dt / p.infectious_period;
  const out = {
    new_unvax: new Int32Array(nReps), new_vax: new Int32Array(nReps),
    first_detect_day: new Float64Array(nReps), last_detect_day: new Float64Array(nReps),
    curves: Array.from({ length: nCurveReps }, () => new Float32Array(nDays)),
  };
  for (let r = 0; r < nReps; r++) {
    let firstDet = -1, lastDet = -1;
    const rng = makeRng(seed + repStart + r);
    const pois = (rate, n) => (rate * n === 0 ? 0 : Math.min(rng.poisson(rate * n), n));
    let Su = N - I0 - sVax0, Sv = sVax0, Eu = 0, Ev = 0, Iu = I0, Iv = 0, nu = 0, nv = 0;
    if (r < nCurveReps) out.curves[r][0] = I0;
    for (let step = 0; step < nSteps; step++) {
      const foi = beta * (Iu + rv * Iv) / N;
      const dSu = pois(foi * dt, Su), dEu = pois(eOut, Eu), dIu = pois(iOut, Iu);
      const dSv = pois(foi * (1 - ve) * dt, Sv), dEv = pois(eOut, Ev), dIv = pois(iOut, Iv);
      Su -= dSu; Eu += dSu - dEu; Iu += dEu - dIu;
      Sv -= dSv; Ev += dSv - dEv; Iv += dEv - dIv;
      nu += dSu; nv += dSv;
      if (dIu + dIv > 0) { if (firstDet < 0) firstDet = step * dt; lastDet = step * dt; }
      if (r < nCurveReps && (step + 1) % stepsPerDay === 0) {
        const day = (step + 1) / stepsPerDay;
        if (day < nDays) out.curves[r][day] = Eu + Ev + Iu + Iv;
      }
      if (Su + Sv === 0 || Eu + Ev + Iu + Iv === 0) break;
    }
    out.new_unvax[r] = nu; out.new_vax[r] = nv;
    out.first_detect_day[r] = firstDet; out.last_detect_day[r] = lastDet;
  }
  return out;
}

function buildSchool(N, nGrades, classSize) {
  const perGrade = Math.ceil(N / nGrades);
  const gradeId = new Int32Array(N), classId = new Int32Array(N);
  let nextClass = 0;
  for (let g = 0; g < nGrades; g++) {
    const idx = [];
    for (let a = 0; a < N; a++) if (Math.floor(a / perGrade) === g) idx.push(a);
    const nCls = Math.max(1, Math.round(idx.length / classSize));
    idx.forEach((a, i) => { gradeId[a] = g; classId[a] = nextClass + (i % nCls); });
    nextClass += nCls;
  }
  const nClass = new Float64Array(nextClass), nGrade = new Float64Array(nGrades);
  const classGrade = new Int32Array(nextClass);
  for (let a = 0; a < N; a++) { nClass[classId[a]]++; nGrade[gradeId[a]]++; classGrade[classId[a]] = gradeId[a]; }
  // members of each class / grade (CSR)
  const startC = new Int32Array(nextClass + 1), startG = new Int32Array(nGrades + 1);
  for (let c = 0; c < nextClass; c++) startC[c + 1] = startC[c] + nClass[c];
  for (let g = 0; g < nGrades; g++) startG[g + 1] = startG[g] + nGrade[g];
  const orderC = Int32Array.from({ length: N }, (_, i) => i).sort((x, y) => classId[x] - classId[y]);
  const orderG = Int32Array.from({ length: N }, (_, i) => i).sort((x, y) => gradeId[x] - gradeId[y]);
  return { classId, gradeId, nClass, nGrade, classGrade, startC, startG, orderC, orderG, nClasses: nextClass };
}

const S = 0, E = 1, I = 2, R = 3;

// Scale of a gamma(shape, scale) truncated to [lo, hi] with truncated mean
// `mean` (same as measles_abm.truncated_gamma_scale).
function truncGammaScale(mean, shape, lo, hi) {
  if (shape === 1 || (lo <= 0 && !Number.isFinite(hi))) return mean / shape;
  const n = 4001, x = new Float64Array(n), lw = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.max(lo, 1e-9) + (hi - Math.max(lo, 1e-9)) * i / (n - 1);
  const tmean = (scale) => {
    let mx = -Infinity;
    for (let i = 0; i < n; i++) { lw[i] = (shape - 1) * Math.log(x[i]) - x[i] / scale; if (lw[i] > mx) mx = lw[i]; }
    let sw = 0, swx = 0;
    for (let i = 0; i < n; i++) { const w = Math.exp(lw[i] - mx); sw += w; swx += w * x[i]; }
    return swx / sw;
  };
  let a = 1e-3 * mean / shape, b = 100 * mean / shape;
  for (let k = 0; k < 100; k++) { const m = 0.5 * (a + b); if (tmean(m) < mean) a = m; else b = m; }
  return 0.5 * (a + b);
}

// Runs reps [repStart, repStart + nReps). Each rep is seeded with seed + rep,
// so splitting reps across Web Workers gives identical results.
function runABM(params, nReps, seed = 12345, repStart = 0, nCurveReps = 0) {
  const p = { ...DEFAULTS, ...params };
  const N = p.population | 0, dt = p.time_step_days;
  const stepsPerDay = Math.round(1 / dt), nSteps = 1 + Math.floor(p.sim_duration_days / dt);
  const I0 = p.I0 | 0, nVax = Math.min(Math.floor(N * p.vax_prop), N - I0);
  const sch = buildSchool(N, p.n_grades, p.class_size);
  const { classId, gradeId, nClass, nGrade, classGrade, startC, startG, orderC, orderG, nClasses } = sch;
  const wsum = p.mixing_weights[0] + p.mixing_weights[1] + p.mixing_weights[2];
  const [wC, wG, wS] = p.mixing_weights.map((w) => w / wsum);
  const beta = p.beta_multiplier * p.R0 / (p.beta_days === null ? p.infectious_period : p.beta_days);
  const incScale = truncGammaScale(p.incubation_period, p.incubation_shape, p.incubation_min_days, p.incubation_max_days);
  const infScale = truncGammaScale(p.infectious_period, p.infectious_shape, p.infectious_min_days, p.infectious_max_days);
  const rashFrac = p.rash_fraction;
  const relInfVax = p.relative_infectiousness_vaccinated, ve = p.vaccine_efficacy;
  const isoEff = p.isolation ? p.isolation_effectiveness : 0;
  const quarantine = p.quarantine, qScope = p.quarantine_scope, qEff = p.quarantine_effectiveness;
  const detectDelay = Math.round(p.detection_delay_days / dt), qSteps = Math.round(p.quarantine_days / dt);
  const nCells = 2 * nClasses;
  const nDays = Math.floor(nSteps / stepsPerDay) + 1;

  const out = {
    new_unvax: new Int32Array(nReps), new_vax: new Int32Array(nReps),
    n_detected: new Int32Array(nReps), quarantine_days: new Float64Array(nReps),
    duration_days: new Float64Array(nReps), n_quarantined: new Int32Array(nReps),
    first_detect_day: new Float64Array(nReps), last_detect_day: new Float64Array(nReps),
    curves: Array.from({ length: nCurveReps }, () => new Float32Array(nDays)),
  };

  // work arrays reused across reps
  const state = new Int8Array(N), vax = new Uint8Array(N), timer = new Int32Array(N);
  const detectAt = new Int32Array(N), isolated = new Uint8Array(N), qUntil = new Int32Array(N);
  const Pc = new Float64Array(nClasses), Pg = new Float64Array(p.n_grades);
  const active = new Int32Array(N), newly = new Int32Array(N), perm = new Int32Array(N);
  const cellStart = new Int32Array(nCells + 1);
  for (let c = 0; c < nClasses; c++) {
    cellStart[2 * c + 1] = cellStart[2 * c] + nClass[c];
    cellStart[2 * c + 2] = cellStart[2 * c + 1] + nClass[c];
  }
  const cellMem = new Int32Array(cellStart[nCells]), cellN = new Int32Array(nCells);

  for (let r = 0; r < nReps; r++) {
    const rng = makeRng(seed + repStart + r);
    const drawSteps = (shape, scale, lo, hi) => {
      if (shape === 1) return rng.geometric(1 - Math.exp(-dt / scale));
      let x = rng.gamma(shape, scale);
      while (x < lo || x > hi) x = rng.gamma(shape, scale);
      const n = Math.floor(x / dt + 0.5);
      return n > 0 ? n : 1;
    };
    const drawInf = () => drawSteps(p.infectious_shape, infScale, p.infectious_min_days, p.infectious_max_days);
    const detectStep = (step, dur) => {
      const d = Math.floor(rashFrac * dur + 0.5) + detectDelay;
      return step + (d < dur - 1 ? d : dur - 1);
    };
    state.fill(S); vax.fill(0); isolated.fill(0); qUntil.fill(-1); cellN.fill(0);
    for (let i = 0; i < N; i++) perm[i] = i;
    for (let i = N - 1; i > 0; i--) { const j = rng.randint(i + 1); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (let k = 0; k < nVax; k++) vax[perm[k]] = 1;
    let nAct = 0;
    for (let k = 0; k < I0; k++) {
      const a = perm[nVax + k];
      state[a] = I; timer[a] = drawInf();
      detectAt[a] = detectStep(0, timer[a]); active[nAct++] = a;
    }
    for (let a = 0; a < N; a++) if (state[a] === S) {
      const cl = 2 * classId[a] + vax[a];
      cellMem[cellStart[cl] + cellN[cl]++] = a;
    }

    if (r < nCurveReps) out.curves[r][0] = I0;
    let step = 0, qDays = 0, nNewU = 0, nNewV = 0, nDet = 0, nQ = 0, firstDet = -1, lastDet = -1;
    while (step < nSteps) {
      Pc.fill(0); Pg.fill(0);
      let Ps = 0;
      for (let i = 0; i < nAct; i++) {
        const a = active[i];
        if (state[a] === I) {
          let x = vax[a] ? relInfVax : 1;
          if (isolated[a]) x *= 1 - isoEff;
          if (qUntil[a] > step) x *= 1 - qEff;
          Pc[classId[a]] += x; Pg[gradeId[a]] += x; Ps += x;
        }
      }

      let nNew = 0;
      if (Ps > 0) {
        const ps = wS * Ps / N;
        for (let c = 0; c < nClasses; c++) {
          const g = classGrade[c];
          const hc = beta * dt * (wC * Pc[c] / nClass[c] + wG * Pg[g] / nGrade[g] + ps);
          for (let v = 0; v < 2; v++) {
            const cl = 2 * c + v, nc = cellN[cl];
            if (nc === 0) continue;
            const h = v ? hc * (1 - ve) : hc;
            if (h <= 0) continue;
            const pr = 1 - Math.exp(-h);
            const k = rng.binomial(nc, pr);
            if (k === 0) continue;
            const pq = 1 - Math.exp(-h * (1 - qEff));
            const base = cellStart[cl];
            for (let j = 0; j < k; j++) {
              const rr = j + rng.randint(nc - j);
              const t = cellMem[base + j]; cellMem[base + j] = cellMem[base + rr]; cellMem[base + rr] = t;
              const a = cellMem[base + j];
              if (qUntil[a] > step && rng.random() * pr >= pq) continue;
              state[a] = E; newly[nNew++] = a;
            }
            for (let j = k - 1; j >= 0; j--) {
              if (state[cellMem[base + j]] === E) { cellN[cl]--; cellMem[base + j] = cellMem[base + cellN[cl]]; }
            }
          }
        }
      }

      let w = 0;
      for (let i = 0; i < nAct; i++) {
        const a = active[i];
        let still = true;
        if (state[a] === E) {
          if (--timer[a] <= 0) {
            state[a] = I; timer[a] = drawInf();
            detectAt[a] = detectStep(step, timer[a]);
          }
        } else {
          if (!isolated[a] && step >= detectAt[a]) {
            isolated[a] = 1; nDet++;
            if (firstDet < 0) firstDet = step * dt;
            lastDet = step * dt;
            if (quarantine) {
              let lo, hi, order;
              if (qScope === 0) { lo = startC[classId[a]]; hi = startC[classId[a] + 1]; order = orderC; }
              else if (qScope === 1) { lo = startG[gradeId[a]]; hi = startG[gradeId[a] + 1]; order = orderG; }
              else { lo = 0; hi = N; order = null; }
              for (let m = lo; m < hi; m++) {
                const b = order ? order[m] : m;
                if (!vax[b] && (state[b] === S || state[b] === E)) {
                  if (qUntil[b] < 0) nQ++;
                  const nu = step + qSteps, prev = qUntil[b] > step ? qUntil[b] : step;
                  qDays += (nu - prev) * dt; qUntil[b] = nu;
                }
              }
            }
          }
          if (--timer[a] <= 0) { state[a] = R; still = false; }
        }
        if (still) active[w++] = a;
      }
      nAct = w;

      for (let k = 0; k < nNew; k++) {
        const a = newly[k];
        timer[a] = drawSteps(p.incubation_shape, incScale, p.incubation_min_days, p.incubation_max_days);
        active[nAct++] = a;
        if (vax[a]) nNewV++; else nNewU++;
      }

      step++;
      if (r < nCurveReps && step % stepsPerDay === 0) out.curves[r][step / stepsPerDay] = nAct;
      if (nAct === 0) break;
    }
    out.new_unvax[r] = nNewU; out.new_vax[r] = nNewV; out.n_detected[r] = nDet;
    out.quarantine_days[r] = qDays; out.duration_days[r] = step * dt;
    out.n_quarantined[r] = nQ; out.first_detect_day[r] = firstDet; out.last_detect_day[r] = lastDet;
  }
  return out;
}

// ---------- Web Worker pool (browser only) ----------
const MODELS = { abm: runABM, seir: runSEIR };
const isWorker = typeof WorkerGlobalScope !== "undefined" && root instanceof WorkerGlobalScope;
const scriptUrl = typeof document !== "undefined" && document.currentScript
  ? document.currentScript.src : "/assets/measles_models.js";

if (isWorker) {
  root.onmessage = (e) => {
    const { id, model, params, nReps, seed, repStart, nCurveReps } = e.data;
    const r = MODELS[model](params, nReps, seed, repStart, nCurveReps);
    root.postMessage({ id, ...r });
  };
}

let pool = [];
let msgId = 0;
function getPool(n) {
  let started = false;
  if (pool.length !== n) {
    pool.forEach((w) => w.terminate());
    pool = Array.from({ length: n }, () => new Worker(scriptUrl));
    started = true;
  }
  return { workers: pool, started };
}
function callWorker(w, msg) {
  return new Promise((resolve, reject) => {
    const id = ++msgId;
    const onMsg = (e) => {
      if (e.data.id !== id) return;
      w.removeEventListener("message", onMsg);
      resolve(e.data);
    };
    w.addEventListener("message", onMsg);
    w.addEventListener("error", reject, { once: true });
    w.postMessage({ id, ...msg });
  });
}
function maxWorkers() {
  const hc = typeof navigator !== "undefined" && navigator.hardwareConcurrency;
  return Math.min(16, hc || 4);
}
// nReps of a model, split across nWorkers Web Workers in contiguous blocks.
// Per-rep seeding makes the result identical to a single-thread run.
async function runParallel(model, params, nReps, seed, nCurveReps, nWorkers = maxWorkers()) {
  const { workers, started } = getPool(nWorkers);
  const per = Math.ceil(nReps / nWorkers);
  const jobs = [];
  for (let w = 0; w < nWorkers; w++) {
    const start = w * per, n = Math.min(per, nReps - start);
    if (n <= 0) break;
    jobs.push(callWorker(workers[w], { model, params, nReps: n, seed, repStart: start,
                                       nCurveReps: w === 0 ? Math.min(nCurveReps, n) : 0 }));
  }
  const parts = await Promise.all(jobs);
  // concatenate every per-simulation typed array (new_unvax, new_vax, ...)
  const out = { curves: parts[0].curves, workersStarted: started };
  for (const key of Object.keys(parts[0])) {
    if (!ArrayBuffer.isView(parts[0][key])) continue;
    const arr = new parts[0][key].constructor(nReps);
    let off = 0;
    for (const part of parts) { arr.set(part[key], off); off += part[key].length; }
    out[key] = arr;
  }
  return out;
}

const API = { DEFAULTS, runABM, runSEIR, runParallel, maxWorkers };
if (typeof module === "object" && module.exports) module.exports = API;
else root.MeaslesModels = API;
})(typeof self !== "undefined" ? self : globalThis);
