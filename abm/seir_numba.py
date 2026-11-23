"""Numba port of the dashboard's measles_efficiency.compute_new_infections
(same algorithm: tau-leap, Poisson capped at compartment size, early stop).
Used for timing comparisons and as the "Compartmental (Numba)" engine of the
test dashboard."""
import numpy as np
from numba import njit


@njit(cache=True)
def _pois(rate, n):
    if rate * n == 0:
        return 0
    return min(np.random.poisson(rate * n), n)


@njit(cache=True)
def seir_simulate(n_reps, seed, N, I0, vax_prop, R0, inc, inf, rel_inf_vax, ve,
                  dt=0.25, days=250, n_curve_reps=0):
    """Returns (new_unvax, new_vax, curves, first_detect_day, last_detect_day).
    curves[rep, day] = E + I at the start of each day for the first
    n_curve_reps reps (as in the dashboard's DashboardExperiment, before the
    7-day moving average). Detection = leaving I (the 5-day infectious period
    ends with isolation); -1 if no case was detected."""
    np.random.seed(seed)
    n_steps = 1 + int(days / dt)
    steps_per_day = int(round(1.0 / dt))
    beta = R0 / inf
    s_vax0 = min(int(N * vax_prop), N - I0)
    out_u = np.zeros(n_reps, np.int64)
    out_v = np.zeros(n_reps, np.int64)
    curves = np.zeros((n_curve_reps, int(days) + 1))
    first_det = np.full(n_reps, -1.0)
    last_det = np.full(n_reps, -1.0)
    for rep in range(n_reps):
        Su, Sv, Eu, Ev, Iu, Iv = N - I0 - s_vax0, s_vax0, 0, 0, I0, 0
        if rep < n_curve_reps:
            curves[rep, 0] = I0
        for step in range(n_steps):
            foi = beta * (Iu + rel_inf_vax * Iv) / N
            dSu = _pois(foi * dt, Su)
            dEu = _pois(dt / inc, Eu)
            dIu = _pois(dt / inf, Iu)
            dSv = _pois(foi * (1 - ve) * dt, Sv)
            dEv = _pois(dt / inc, Ev)
            dIv = _pois(dt / inf, Iv)
            Su -= dSu; Eu += dSu - dEu; Iu += dEu - dIu
            Sv -= dSv; Ev += dSv - dEv; Iv += dEv - dIv
            out_u[rep] += dSu; out_v[rep] += dSv
            if dIu + dIv > 0:
                if first_det[rep] < 0.0:
                    first_det[rep] = step * dt
                last_det[rep] = step * dt
            if rep < n_curve_reps and (step + 1) % steps_per_day == 0:
                day = (step + 1) // steps_per_day
                if day <= days:
                    curves[rep, day] = Eu + Ev + Iu + Iv
            if Su + Sv == 0 or Eu + Ev + Iu + Iv == 0:
                break
    return out_u, out_v, curves, first_det, last_det


@njit(cache=True)
def seir_new_infections(n_reps, seed, N, I0, vax_prop, R0, inc, inf, rel_inf_vax, ve,
                        dt=0.25, days=250):
    out_u, out_v, _, _, _ = seir_simulate(n_reps, seed, N, I0, vax_prop, R0, inc, inf,
                                    rel_inf_vax, ve, dt, days, 0)
    return out_u, out_v
