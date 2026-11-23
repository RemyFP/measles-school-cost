"""Check that the fast kernels reproduce the reference kernel's statistics."""
import time
import numpy as np
from measles_abm import run_abm

BASE = {"mixing_weights": (0.5, 0.2, 0.3), "beta_multiplier": 1.22,
        "incubation_shape": 4.0, "infectious_shape": 4.0,
        "rash_fraction": 0.0, "detection_delay_days": 4.0, "isolation_effectiveness": 0.9, }
CASES = {
    "well-mixed 85%": {"vax_prop": 0.85},
    "structured 90%": {**BASE, "vax_prop": 0.90},
    "structured 85% iso+class quarantine": {**BASE, "vax_prop": 0.85, "isolation": True, "quarantine": True},
    "structured 85% iso+school quarantine": {**BASE, "vax_prop": 0.85, "isolation": True, "quarantine": True,
                                             "quarantine_scope": 2},
}
for alg in ("reference", "active", "group"):
    run_abm({}, 2, algorithm=alg); run_abm({}, 2, algorithm=alg, parallel=True)

n = 20000
for name, prm in CASES.items():
    print(name)
    for alg in ("reference", "active", "group"):
        r = run_abm(prm, n, algorithm=alg, parallel=True)
        t = r["new_unvax"] + r["new_vax"]
        se = t.std() / np.sqrt(n)
        print(f"   {alg:<9} P(>=10)={np.mean(t >= 10):.3f}  mean new={t.mean():6.2f} (±{1.96*se:.2f})  "
              f"mean vax cases={r['new_vax'].mean():.3f}  quarantine days={r['quarantine_days'].mean():7.1f}")
