"""R7b (2026-10-13): the R7 parameter-inflation fix applied to the remaining estimation rows of Paper II's
model-error table, so every 'est.' row uses the same remedy (R7 fixed only the three rows the latent filter won,
plus J17G). Same grid as R7 restricted to spread floor 0.02 (0.05 never won any of the six R7 cases), state RTPS
r and r - 0.2; seeds 1-2 at each case's best seed-0 setting. Three runs at a time, 2 threads each. Log: r7.log.
"""
from concurrent.futures import ThreadPoolExecutor
import json

import run_r7 as r7

r7.BASES = ["J16a_lap1.1_rtps0.7_s0", "J17H_p5_c16_rtps0.7_abg_s0", "J17H_p6_c16_rtps0.7_abg_s0",
            "J17B_e0.1_pde_c8.25_rtps0.7_lap_s0", "J17B_e0.2_pde_c8.25_rtps0.7_lap_s0"]


def main():
    jobs = []
    for b in r7.BASES:
        r0 = json.load(open(r7.E / f"i11_{b}.json"))["args"]["rtps"]
        jobs += [(b, 0.02, r, 0) for r in (r0, round(r0 - 0.2, 2))]
    r7.log(f"R7b start: {len(jobs)} seed-0 runs")
    with ThreadPoolExecutor(3) as ex:
        tags = list(ex.map(lambda j: r7.run(*j), jobs))
    best = {}
    for j, t in zip(jobs, tags):
        if j[0] not in best or r7.steady(t) < r7.steady(best[j[0]][1]):
            best[j[0]] = (j, t)
    for b, (j, t) in best.items():
        orig = json.load(open(r7.E / f"i11_{b}.json"))["steady"]["analysis"]
        r7.log(f"R7b best seed 0 for {b}: {t} ({r7.steady(t):.3f}; original {orig:.3f})")
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda j: r7.run(*j), [(j[0], j[1], j[2], s) for j, _ in best.values() for s in (1, 2)]))
    r7.log("R7b DONE")


if __name__ == "__main__":
    main()
