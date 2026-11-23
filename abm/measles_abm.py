"""
Simple agent-based / group-network measles model for a single school.

Design goals
------------
1. Reproduce the dashboard SEIR (measles-dashboard/measles_efficiency.py) when
   mixing is set to "well mixed" and periods are exponential -> validation.
2. Add school structure (classrooms within grades + school-wide mixing) and the
   things the compartmental model cannot do well: isolation of detected cases
   and quarantine (exclusion) of unvaccinated contacts, with per-student
   tracking of days absent (directly usable in a cost model).
3. Be fast enough for an interactive tool (Numba-compiled, early stopping).

Transmission is "group based": instead of an explicit edge list, each student
belongs to a classroom, a grade and the school. The hazard for a susceptible
student j at a time step is

    h_j = beta * dt * sus_j * [ w_class  * P_class(c_j) / n_class(c_j)
                              + w_grade  * P_grade(g_j) / n_grade(g_j)
                              + w_school * P_school     / N ]

where P_x is the infectiousness-weighted number of infectious students in the
group (vaccinated infectious weighted by `rel_inf_vax`, isolated / quarantined
students weighted down). With w_school = 1 this is exactly the frequency-
dependent force of infection of the dashboard SEIR (beta = R0 / D), so each
infectious student generates R0 infections in a fully susceptible school
regardless of the mixing weights (up to finite-group effects).

Group-based mixing is equivalent to a weighted complete graph inside each
group, costs O(N) per step (no edges stored) and supports quarantine of
classmates trivially.
"""

import time
import types
import numpy as np
from numba import njit, prange

# Compartment codes
S, E, I, R = 0, 1, 2, 3


DEFAULT_ABM_PARAMS = {
    # --- same meaning as dashboard parameters ---
    "R0": 15.0,
    "incubation_period": 10.5,          # mean days E
    "infectious_period": 5.0,           # mean days I
    "relative_infectiousness_vaccinated": 0.05,
    "vaccine_efficacy": 0.997,
    "population": 500,
    "I0": 1,
    "vax_prop": 0.85,
    "sim_duration_days": 250,
    "time_step_days": 0.25,
    # --- ABM-specific: school structure ---
    "n_grades": 6,                      # e.g. K-5 elementary
    "class_size": 22,
    # mixing weights (class, grade, school); (0, 0, 1) == dashboard SEIR
    "mixing_weights": (0.0, 0.0, 1.0),
    "beta_multiplier": 1.0,             # calibration knob for structured mixing
    # Period distributions: shape=1 -> exponential (dashboard SEIR), otherwise
    # gamma with the given mean, truncated to [min, max] (redrawn outside).
    "incubation_shape": 1.0,
    "incubation_min_days": 0.0,
    "incubation_max_days": np.inf,
    "infectious_shape": 1.0,
    "infectious_min_days": 0.0,
    "infectious_max_days": np.inf,
    # Days over which R0 is defined: beta = R0 / beta_days. None -> the mean
    # infectious period (dashboard SEIR). With isolation, R0 refers to the
    # infectious days before isolation (dashboard: 5 = rash on day 4 of 8 + 1).
    "beta_days": None,
    # --- detection (rash) and interventions ---
    # A case is detected (isolated, contacts quarantined) at
    #   infectiousness onset + rash_fraction * own infectious period + detection_delay_days,
    # or on its last infectious day if that comes first. rash_fraction=1 and
    # delay=0 -> detection when infectiousness ends (dashboard SEIR, where the
    # 5-day infectious period already ends with isolation).
    "rash_fraction": 1.0,
    "detection_delay_days": 0.0,
    "isolation": False,
    "isolation_effectiveness": 1.0,     # reduction in transmission once isolated
    "quarantine": False,
    "quarantine_scope": 0,              # 0 = classmates, 1 = grade, 2 = whole school
    "quarantine_days": 21.0,
    "quarantine_effectiveness": 0.9,    # reduction in school contacts in quarantine
}


def build_school(N: int, n_grades: int, class_size: int):
    """Assign students to grades (equal split) and classrooms within grades."""
    grade_id = np.repeat(np.arange(n_grades), int(np.ceil(N / n_grades)))[:N]
    class_id = np.empty(N, dtype=np.int64)
    next_class = 0
    for g in range(n_grades):
        idx = np.where(grade_id == g)[0]
        n_cls = max(1, int(round(len(idx) / class_size)))
        class_id[idx] = next_class + (np.arange(len(idx)) % n_cls)
        next_class += n_cls
    n_class = np.bincount(class_id).astype(np.float64)
    n_grade = np.bincount(grade_id).astype(np.float64)
    return class_id, grade_id.astype(np.int64), n_class, n_grade


def truncated_gamma_scale(mean, shape, lo, hi):
    """Scale of a gamma(shape, scale) truncated to [lo, hi] whose truncated
    mean is `mean` (bisection; grid integration of the density)."""
    if shape == 1.0 or (lo <= 0.0 and not np.isfinite(hi)):
        return mean / shape
    if not lo < mean < hi:
        raise ValueError(f"mean {mean} outside truncation bounds [{lo}, {hi}]")
    x = np.linspace(max(lo, 1e-9), hi, 4001)

    def tmean(scale):
        logw = (shape - 1.0) * np.log(x) - x / scale
        w = np.exp(logw - logw.max())
        return float(np.sum(w * x) / np.sum(w))

    a, b = 1e-3 * mean / shape, 100.0 * mean / shape
    for _ in range(100):
        m = 0.5 * (a + b)
        if tmean(m) < mean:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


@njit(cache=True)
def _draw_steps(shape, scale, lo, hi, dt):
    """Duration in time steps. shape==1: geometric with mean `scale` days (the
    SEIR's per-step exit probability); otherwise a discretised gamma truncated
    to [lo, hi] days by redrawing."""
    if shape == 1.0:
        p = 1.0 - np.exp(-dt / scale)
        return np.random.geometric(p)
    x = np.random.gamma(shape, scale)
    while x < lo or x > hi:
        x = np.random.gamma(shape, scale)
    n = int(x / dt + 0.5)
    return n if n > 0 else 1


@njit(cache=True)
def _detect_step(step, dur, rash_frac, detect_delay_steps):
    """Step at which a case that became infectious at `step`, for `dur` steps,
    is detected (never after its last infectious step)."""
    d = int(rash_frac * dur + 0.5) + detect_delay_steps
    return step + (d if d < dur - 1 else dur - 1)


def _simulate_py(n_reps, seed, N, n_vax, I0, beta, rel_inf_vax, ve,
              inc_shape, inc_scale, inc_lo, inc_hi, inf_shape, inf_scale, inf_lo, inf_hi,
              class_id, grade_id, n_class, n_grade, w_class, w_grade, w_school,
              dt, n_steps, steps_per_day,
              rash_frac, detect_delay_steps, iso_eff,
              quarantine, q_scope, q_steps, q_eff,
              n_curve_reps):

    n_classes = n_class.shape[0]
    n_grades = n_grade.shape[0]
    n_days = n_steps // steps_per_day + 1

    # outputs
    new_unvax = np.zeros(n_reps, np.int64)
    new_vax = np.zeros(n_reps, np.int64)
    n_detected = np.zeros(n_reps, np.int64)
    quarantine_days = np.zeros(n_reps)      # student-days in quarantine
    n_quarantined = np.zeros(n_reps, np.int64)  # students quarantined at least once
    first_detect = np.full(n_reps, -1.0)    # day of first / last detection
    last_detect = np.full(n_reps, -1.0)
    duration_days = np.zeros(n_reps)
    curves = np.zeros((n_curve_reps, n_days))  # E+I at end of each day

    # members of each class / grade (CSR) for quarantine lookups
    order_c = np.argsort(class_id)
    start_c = np.zeros(n_classes + 1, np.int64)
    for c in range(n_classes):
        start_c[c + 1] = start_c[c] + int(n_class[c])
    order_g = np.argsort(grade_id)
    start_g = np.zeros(n_grades + 1, np.int64)
    for g in range(n_grades):
        start_g[g + 1] = start_g[g] + int(n_grade[g])

    for rep in prange(n_reps):
        # Per-rep seeding + per-rep work arrays: results are identical whether
        # reps run serially or in parallel threads, and don't depend on n_reps.
        np.random.seed(seed + rep)
        state = np.zeros(N, np.int8)
        vax = np.zeros(N, np.bool_)
        timer = np.zeros(N, np.int64)
        detect_at = np.zeros(N, np.int64)
        isolated = np.zeros(N, np.bool_)
        q_until = np.zeros(N, np.int64)
        P_class = np.zeros(n_classes)
        P_grade = np.zeros(n_grades)
        newly = np.empty(N, np.int64)

        # ---- initialise ----
        perm = np.random.permutation(N)
        state[:] = S
        vax[:] = False
        isolated[:] = False
        q_until[:] = -1
        timer[:] = 0
        for k in range(n_vax):
            vax[perm[k]] = True
        for k in range(I0):
            a = perm[n_vax + k]          # index cases are unvaccinated
            state[a] = I
            timer[a] = _draw_steps(inf_shape, inf_scale, inf_lo, inf_hi, dt)
            detect_at[a] = _detect_step(0, timer[a], rash_frac, detect_delay_steps)
        n_active = I0

        step = 0
        while step < n_steps:
            # ---- infectious pressure per group ----
            P_class[:] = 0.0
            P_grade[:] = 0.0
            P_school = 0.0
            for a in range(N):
                if state[a] == I:
                    x = rel_inf_vax if vax[a] else 1.0
                    if isolated[a]:
                        x *= (1.0 - iso_eff)
                    if q_until[a] > step:
                        x *= (1.0 - q_eff)
                    P_class[class_id[a]] += x
                    P_grade[grade_id[a]] += x
                    P_school += x

            # ---- infections (based on state at start of step) ----
            n_new = 0
            if P_school > 0.0:
                ps = w_school * P_school / N
                for a in range(N):
                    if state[a] == S:
                        h = beta * dt * (w_class * P_class[class_id[a]] / n_class[class_id[a]]
                                         + w_grade * P_grade[grade_id[a]] / n_grade[grade_id[a]]
                                         + ps)
                        if vax[a]:
                            h *= (1.0 - ve)
                        if q_until[a] > step:
                            h *= (1.0 - q_eff)
                        if h > 0.0 and np.random.random() < 1.0 - np.exp(-h):
                            newly[n_new] = a
                            n_new += 1

            # ---- progression of existing E / I ----
            for a in range(N):
                st = state[a]
                if st == E:
                    timer[a] -= 1
                    if timer[a] <= 0:
                        state[a] = I
                        timer[a] = _draw_steps(inf_shape, inf_scale, inf_lo, inf_hi, dt)
                        detect_at[a] = _detect_step(step, timer[a], rash_frac,
                                                    detect_delay_steps)
                elif st == I:
                    # detection (rash) -> isolation and quarantine of contacts
                    if (not isolated[a]) and step >= detect_at[a]:
                        # `isolated` doubles as the "detected" flag; when
                        # isolation is off, iso_eff is passed as 0
                        isolated[a] = True
                        n_detected[rep] += 1
                        if first_detect[rep] < 0.0:
                            first_detect[rep] = step * dt
                        last_detect[rep] = step * dt
                        if quarantine:
                            if q_scope == 0:
                                lo, hi = start_c[class_id[a]], start_c[class_id[a] + 1]
                                members = order_c[lo:hi]
                            elif q_scope == 1:
                                lo, hi = start_g[grade_id[a]], start_g[grade_id[a] + 1]
                                members = order_g[lo:hi]
                            else:
                                members = np.arange(N)
                            for b in members:
                                # exclude unvaccinated students without
                                # known disease (S, or E: not yet detected)
                                if (not vax[b]) and (state[b] == S or state[b] == E):
                                    if q_until[b] < 0:
                                        n_quarantined[rep] += 1
                                    q_until[b] = step + q_steps
                    timer[a] -= 1
                    if timer[a] <= 0:
                        state[a] = R
                        n_active -= 1

            for k in range(n_new):
                a = newly[k]
                state[a] = E
                timer[a] = _draw_steps(inc_shape, inc_scale, inc_lo, inc_hi, dt)
                n_active += 1
                if vax[a]:
                    new_vax[rep] += 1
                else:
                    new_unvax[rep] += 1

            # quarantine bookkeeping (student-days excluded)
            if quarantine:
                nq = 0
                for a in range(N):
                    if q_until[a] > step:
                        nq += 1
                quarantine_days[rep] += nq * dt

            step += 1
            if rep < n_curve_reps and step % steps_per_day == 0:
                cnt = 0
                for a in range(N):
                    if state[a] == E or state[a] == I:
                        cnt += 1
                curves[rep, step // steps_per_day] = cnt

            if n_active == 0:
                # finish remaining quarantine days analytically
                if quarantine:
                    for a in range(N):
                        if q_until[a] > step:
                            quarantine_days[rep] += (q_until[a] - step) * dt
                break

        duration_days[rep] = step * dt

    return (new_unvax, new_vax, n_detected, quarantine_days, duration_days, curves,
            n_quarantined, first_detect, last_detect)


def _simulate_fast_py(n_reps, seed, N, n_vax, I0, beta, rel_inf_vax, ve,
                      inc_shape, inc_scale, inc_lo, inc_hi, inf_shape, inf_scale, inf_lo, inf_hi,
                      class_id, grade_id, n_class, n_grade, w_class, w_grade, w_school,
                      dt, n_steps, steps_per_day,
                      rash_frac, detect_delay_steps, iso_eff,
                      quarantine, q_scope, q_steps, q_eff,
                      n_curve_reps, group_draws):
    """Same model as `_simulate_py`, but per-step work does not scale with N.

    1. Active list: infectious pressure and E/I progression loop only over
       students currently in E or I.
    2. Per-group draws (group_draws=True): susceptibles are kept in "cells"
       (classroom x vaccinated). Everyone in a cell has the same hazard, so the
       number infected is Binomial(n_cell, p_cell) and the infected students
       are a uniformly random subset (partial Fisher-Yates shuffle). This is
       exactly equivalent to independent Bernoulli draws per student.
       Quarantined students have a lower hazard: they are handled by thinning
       (a selected quarantined student is infected with prob. p_q / p).
       With group_draws=False, susceptibles are still checked one by one.
    """
    n_classes = n_class.shape[0]
    n_grades = n_grade.shape[0]
    n_days = n_steps // steps_per_day + 1
    n_cells = 2 * n_classes                 # cell = 2*class + vaccinated

    new_unvax = np.zeros(n_reps, np.int64)
    new_vax = np.zeros(n_reps, np.int64)
    n_detected = np.zeros(n_reps, np.int64)
    quarantine_days = np.zeros(n_reps)
    n_quarantined = np.zeros(n_reps, np.int64)
    first_detect = np.full(n_reps, -1.0)
    last_detect = np.full(n_reps, -1.0)
    duration_days = np.zeros(n_reps)
    curves = np.zeros((n_curve_reps, n_days))

    order_c = np.argsort(class_id)
    start_c = np.zeros(n_classes + 1, np.int64)
    for c in range(n_classes):
        start_c[c + 1] = start_c[c] + int(n_class[c])
    order_g = np.argsort(grade_id)
    start_g = np.zeros(n_grades + 1, np.int64)
    for g in range(n_grades):
        start_g[g + 1] = start_g[g] + int(n_grade[g])
    class_grade = np.zeros(n_classes, np.int64)
    for a in range(N):
        class_grade[class_id[a]] = grade_id[a]

    for rep in prange(n_reps):
        np.random.seed(seed + rep)
        state = np.zeros(N, np.int8)
        vax = np.zeros(N, np.bool_)
        timer = np.zeros(N, np.int64)
        detect_at = np.zeros(N, np.int64)
        isolated = np.zeros(N, np.bool_)
        q_until = np.full(N, -1, np.int64)
        P_class = np.zeros(n_classes)
        P_grade = np.zeros(n_grades)
        active = np.empty(N, np.int64)
        newly = np.empty(N, np.int64)

        perm = np.random.permutation(N)
        for k in range(n_vax):
            vax[perm[k]] = True
        n_act = 0
        for k in range(I0):
            a = perm[n_vax + k]
            state[a] = I
            timer[a] = _draw_steps(inf_shape, inf_scale, inf_lo, inf_hi, dt)
            detect_at[a] = _detect_step(0, timer[a], rash_frac, detect_delay_steps)
            active[n_act] = a
            n_act += 1

        # susceptible cells (CSR, capacity = class size; cell_n = current count)
        cell_start = np.zeros(n_cells + 1, np.int64)
        for c in range(n_classes):
            cell_start[2 * c + 1] = cell_start[2 * c] + int(n_class[c])
            cell_start[2 * c + 2] = cell_start[2 * c + 1] + int(n_class[c])
        cell_mem = np.empty(cell_start[n_cells], np.int64)
        cell_n = np.zeros(n_cells, np.int64)
        if group_draws:
            for a in range(N):
                if state[a] == S:
                    cl = 2 * class_id[a] + (1 if vax[a] else 0)
                    cell_mem[cell_start[cl] + cell_n[cl]] = a
                    cell_n[cl] += 1

        step = 0
        while step < n_steps:
            # ---- infectious pressure (active list only) ----
            P_class[:] = 0.0
            P_grade[:] = 0.0
            P_school = 0.0
            for i in range(n_act):
                a = active[i]
                if state[a] == I:
                    x = rel_inf_vax if vax[a] else 1.0
                    if isolated[a]:
                        x *= (1.0 - iso_eff)
                    if q_until[a] > step:
                        x *= (1.0 - q_eff)
                    P_class[class_id[a]] += x
                    P_grade[grade_id[a]] += x
                    P_school += x

            # ---- infections ----
            n_new = 0
            if P_school > 0.0:
                ps = w_school * P_school / N
                if group_draws:
                    for c in range(n_classes):
                        g = class_grade[c]
                        h_c = beta * dt * (w_class * P_class[c] / n_class[c]
                                           + w_grade * P_grade[g] / n_grade[g] + ps)
                        for v in range(2):
                            cl = 2 * c + v
                            n_c = cell_n[cl]
                            if n_c == 0:
                                continue
                            h = h_c * (1.0 - ve) if v == 1 else h_c
                            if h <= 0.0:
                                continue
                            p = 1.0 - np.exp(-h)
                            k = np.random.binomial(n_c, p)
                            if k == 0:
                                continue
                            p_q = 1.0 - np.exp(-h * (1.0 - q_eff))
                            base = cell_start[cl]
                            # partial Fisher-Yates: first k slots = random subset
                            for j in range(k):
                                r = j + np.random.randint(n_c - j)
                                tmp = cell_mem[base + j]
                                cell_mem[base + j] = cell_mem[base + r]
                                cell_mem[base + r] = tmp
                                a = cell_mem[base + j]
                                if q_until[a] > step and np.random.random() * p >= p_q:
                                    continue          # thinned out (quarantined)
                                state[a] = E
                                newly[n_new] = a
                                n_new += 1
                            # remove infected from the cell (swap with tail)
                            for j in range(k - 1, -1, -1):
                                if state[cell_mem[base + j]] == E:
                                    cell_n[cl] -= 1
                                    cell_mem[base + j] = cell_mem[base + cell_n[cl]]
                else:
                    for a in range(N):
                        if state[a] == S:
                            h = beta * dt * (w_class * P_class[class_id[a]] / n_class[class_id[a]]
                                             + w_grade * P_grade[grade_id[a]] / n_grade[grade_id[a]]
                                             + ps)
                            if vax[a]:
                                h *= (1.0 - ve)
                            if q_until[a] > step:
                                h *= (1.0 - q_eff)
                            if h > 0.0 and np.random.random() < 1.0 - np.exp(-h):
                                newly[n_new] = a
                                n_new += 1
                    for k2 in range(n_new):
                        state[newly[k2]] = E

            # ---- progression of students that were E/I at start of step ----
            w_ptr = 0
            for i in range(n_act):
                a = active[i]
                still = True
                if state[a] == E:
                    timer[a] -= 1
                    if timer[a] <= 0:
                        state[a] = I
                        timer[a] = _draw_steps(inf_shape, inf_scale, inf_lo, inf_hi, dt)
                        detect_at[a] = _detect_step(step, timer[a], rash_frac,
                                                    detect_delay_steps)
                else:  # I
                    if (not isolated[a]) and step >= detect_at[a]:
                        isolated[a] = True
                        n_detected[rep] += 1
                        if first_detect[rep] < 0.0:
                            first_detect[rep] = step * dt
                        last_detect[rep] = step * dt
                        if quarantine:
                            if q_scope == 0:
                                members = order_c[start_c[class_id[a]]:start_c[class_id[a] + 1]]
                            elif q_scope == 1:
                                members = order_g[start_g[grade_id[a]]:start_g[grade_id[a] + 1]]
                            else:
                                members = np.arange(N)
                            for b in members:
                                if (not vax[b]) and (state[b] == S or state[b] == E):
                                    if q_until[b] < 0:
                                        n_quarantined[rep] += 1
                                    new_until = step + q_steps
                                    prev = q_until[b] if q_until[b] > step else step
                                    quarantine_days[rep] += (new_until - prev) * dt
                                    q_until[b] = new_until
                    timer[a] -= 1
                    if timer[a] <= 0:
                        state[a] = R
                        still = False
                if still:
                    active[w_ptr] = a
                    w_ptr += 1
            n_act = w_ptr

            # ---- newly infected join the active list ----
            for k2 in range(n_new):
                a = newly[k2]
                timer[a] = _draw_steps(inc_shape, inc_scale, inc_lo, inc_hi, dt)
                active[n_act] = a
                n_act += 1
                if vax[a]:
                    new_vax[rep] += 1
                else:
                    new_unvax[rep] += 1

            step += 1
            if rep < n_curve_reps and step % steps_per_day == 0:
                curves[rep, step // steps_per_day] = n_act
            if n_act == 0:
                break

        duration_days[rep] = step * dt

    return (new_unvax, new_vax, n_detected, quarantine_days, duration_days, curves,
            n_quarantined, first_detect, last_detect)


def _renamed_copy(fn, name):
    """Copy of `fn` under another name: Numba's on-disk cache is keyed by the
    function's qualified name, so the serial and parallel builds of the same
    Python function would otherwise overwrite / load each other's cache."""
    code = fn.__code__.replace(co_name=name, co_qualname=name)
    new = types.FunctionType(code, fn.__globals__, name, fn.__defaults__)
    new.__qualname__ = name
    return new


_simulate_serial = njit(cache=True)(_renamed_copy(_simulate_py, "_simulate_serial_py"))
_simulate_parallel = njit(cache=True, parallel=True)(
    _renamed_copy(_simulate_py, "_simulate_parallel_py"))
_simulate_fast_serial = njit(cache=True)(_renamed_copy(_simulate_fast_py, "_simulate_fast_serial_py"))
_simulate_fast_parallel = njit(cache=True, parallel=True)(
    _renamed_copy(_simulate_fast_py, "_simulate_fast_parallel_py"))


def _isolation_flag_fix(params):
    # When quarantine is on but isolation is off we reuse `isolated` as a
    # "detected" flag; neutralise its effect on transmission.
    return params["isolation_effectiveness"] if params["isolation"] else 0.0


def run_abm(params: dict, n_reps: int, seed: int = 147125098488,
            n_curve_reps: int = 0, parallel: bool = False,
            algorithm: str = "group"):
    """algorithm: "reference" (loop over all students, original version),
    "active" (active list, per-student susceptible draws) or
    "group" (active list + per-group binomial draws; fastest)."""
    p = {**DEFAULT_ABM_PARAMS, **params}
    N = int(p["population"])
    dt = float(p["time_step_days"])
    steps_per_day = int(round(1.0 / dt))
    n_steps = 1 + int(p["sim_duration_days"] / dt)
    n_vax = min(int(N * p["vax_prop"]), N - int(p["I0"]))
    class_id, grade_id, n_class, n_grade = build_school(
        N, int(p["n_grades"]), int(p["class_size"]))
    w = np.asarray(p["mixing_weights"], dtype=float)
    w = w / w.sum()
    beta_days = p["infectious_period"] if p["beta_days"] is None else p["beta_days"]
    beta = p["beta_multiplier"] * p["R0"] / beta_days
    inc = (float(p["incubation_shape"]),
           truncated_gamma_scale(float(p["incubation_period"]), float(p["incubation_shape"]),
                                 float(p["incubation_min_days"]), float(p["incubation_max_days"])),
           float(p["incubation_min_days"]), float(p["incubation_max_days"]))
    inf = (float(p["infectious_shape"]),
           truncated_gamma_scale(float(p["infectious_period"]), float(p["infectious_shape"]),
                                 float(p["infectious_min_days"]), float(p["infectious_max_days"])),
           float(p["infectious_min_days"]), float(p["infectious_max_days"]))

    if algorithm == "reference":
        sim = _simulate_parallel if parallel else _simulate_serial
        extra = ()
    else:
        sim = _simulate_fast_parallel if parallel else _simulate_fast_serial
        extra = (algorithm == "group",)
    out = sim(
        n_reps, seed % (2**31), N, n_vax, int(p["I0"]), beta,
        p["relative_infectiousness_vaccinated"], p["vaccine_efficacy"],
        *inc, *inf,
        class_id, grade_id, n_class, n_grade, w[0], w[1], w[2],
        dt, n_steps, steps_per_day,
        float(p["rash_fraction"]),
        int(round(p["detection_delay_days"] / dt)), _isolation_flag_fix(p),
        bool(p["quarantine"]), int(p["quarantine_scope"]),
        int(round(p["quarantine_days"] / dt)), p["quarantine_effectiveness"],
        n_curve_reps, *extra)
    keys = ["new_unvax", "new_vax", "n_detected", "quarantine_days",
            "duration_days", "curves", "n_quarantined", "first_detect_day",
            "last_detect_day"]
    return dict(zip(keys, out))


if __name__ == "__main__":
    t = time.perf_counter()
    run_abm({}, 2)
    print(f"JIT compile / cache load: {time.perf_counter() - t:.2f}s")
    t = time.perf_counter()
    res = run_abm({}, 1000)
    tot = res["new_unvax"] + res["new_vax"]
    print(f"1000 reps: {time.perf_counter() - t:.3f}s, "
          f"P(>=10 new)={np.mean(tot >= 10):.3f}, mean new={tot.mean():.1f}")
