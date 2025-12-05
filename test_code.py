# test.py
from __future__ import annotations

import os
import time
import random
from typing import List, Tuple, Optional, Dict
import statistics
import math
from core.env import Env
from planner.cbs import CBSPlanner
from planner.prioritize import PrioritizedPlanner
from planner.evaluation import Analyzer, PlannerRun, create_animation



Pos = Tuple[int, int]

def sample_random_goals(env: Env, seed: int = 0) -> List[Pos]:
    '''
    sample n distinct random goal positions for robots in the env
    '''
    rng = random.Random(seed)
    n = env.robots_number
    goals: List[Pos] = []
    used = set()

    mid = (env.w // 2, env.h // 2)

    attempts = 0
    while len(goals) < n:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("Failed to sample enough goals; map too constrained.")

        x = rng.randrange(0, env.w)
        y = rng.randrange(0, env.h)
        if env.is_obstacle(x, y):
            continue
        if (x, y) in used:
            continue
        if (x, y) == mid:
            continue

        used.add((x, y))
        goals.append((x, y))
    return goals


def mean_std_ignore_nan(values: List[float]) -> Tuple[float, float, int]:
    """Return (mean, std, count) ignoring NaN/inf."""
    xs = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not xs:
        return float("nan"), float("nan"), 0
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) >= 2 else 0.0  # sample std
    return mu, sd, len(xs)


def run_pp(env: Env, goals: List[Pos], priority_mode: str = "default", run_seed: Optional[int] = None) -> PlannerRun:
    # Make "random" mode reproducible (and also stabilize any other internal randomness).
    if run_seed is not None:
        random.seed(run_seed)

    planner = PrioritizedPlanner(env, priority_mode=priority_mode)
    t0 = time.perf_counter()
    paths = planner.process(goals)
    dt = time.perf_counter() - t0
    planner_name = f"PP-{priority_mode}"
    return PlannerRun(
        planner=planner_name, map_name="", n_robots=env.robots_number, n_obstacles=len(env.obstacles),
        success=bool(paths), runtime_s=dt, paths=paths if paths else []
    )


def run_cbs(env: Env, goals: List[Pos], max_time: int = 200) -> PlannerRun:
    planner = CBSPlanner(env, max_time=max_time)
    t0 = time.perf_counter()
    paths = planner.process(goals)
    dt = time.perf_counter() - t0
    return PlannerRun(
        planner="CBS", map_name="", n_robots=env.robots_number, n_obstacles=len(env.obstacles),
        success=bool(paths), runtime_s=dt, paths=paths if paths else []
    )


def create_table(map_path: str, out_dir: str = "results") -> None:
    env = Env(map_path)
    #goals = [(36,4),(5,45),(45,14),(24,35),(36,21),(7,35),(5,35),(2,25)]
    goals = sample_random_goals(env, seed=123)

    r_pp_default = run_pp(env, goals, priority_mode="default")
    #create_animation(env, r_pp_default.paths, file_name="results/pp_default_animation.gif", interval=200, enabled=True)
    
    r_pp_random = run_pp(env, goals, priority_mode="random")
    create_animation(env, r_pp_random.paths, file_name="results/pp_random_animation.gif", interval=200, enabled=True)
    
    r_pp_closest = run_pp(env, goals, priority_mode="closest")
    #create_animation(env, r_pp_closest.paths, file_name="results/pp_closest_animation.gif", interval=200, enabled=True)
    
    r_pp_far = run_pp(env, goals, priority_mode="far")
    #create_animation(env, r_pp_far.paths, file_name="results/pp_far_animation.gif", interval=200, enabled=True)
    #print("running cbs...")
    #r_cbs = run_cbs(env, goals, max_time=200)

    #create_animation(env, r_cbs.paths, file_name="results/cbs_animation.gif", interval=200, enabled=True)
    map_name = os.path.basename(map_path)
    for r in ([r_pp_default, r_pp_random, r_pp_closest, r_pp_far]): #r_cbs]):
        r.map_name = map_name
    # for r in ([r_pp_far,r_cbs]): #r_cbs]):
    #     r.map_name = map_name

    rows = [Analyzer.compute_metrics(r_pp_default),Analyzer.compute_metrics(r_pp_random),Analyzer.compute_metrics(r_pp_closest),Analyzer.compute_metrics(r_pp_far)]#, Analyzer.compute_metrics(r_cbs)]
    #rows = [Analyzer.compute_metrics(r_pp_far), Analyzer.compute_metrics(r_cbs)]#, Analyzer.compute_metrics(r_cbs)]

    Analyzer.save_csv(rows, f"{out_dir}/presentation_fixed_case.csv")
    Analyzer.save_xlsx(
        rows,
        f"{out_dir}/presentation_fixed_case.xlsx",
        sheet_name="results",
        also_save_transposed=True,
        transpose_sheet_name="transpose",
    )


    #print("[OK] fixed case -> results/presentation_fixed_case.(csv/md)")


def sweep_runtime_by_map(
    map_paths: List[str],
    x_label: str,
    x_values: List[int],
    out_png: str,
    out_csv: str,
    out_dir: str = "results",
    cbs_max_time: int = 200,
    n_seeds: int = 10,
    seed0: int = 300,
) -> None:
    """
    For each map (same map layout for that x), run n_seeds different goal samplings.
    Compute mean/std of the metric and plot mean ± std.
    """
    all_rows = []

    # mean curves (what you plot)
    pp_default_mean: List[float] = []
    pp_random_mean: List[float] = []
    pp_closest_mean: List[float] = []
    pp_far_mean: List[float] = []

    # std curves (error bars)
    pp_default_std: List[float] = []
    pp_random_std: List[float] = []
    pp_closest_std: List[float] = []
    pp_far_std: List[float] = []

    for map_path, x in zip(map_paths, x_values):
        env = Env(map_path)
        map_name = os.path.basename(map_path)

        default_vals: List[float] = []
        random_vals: List[float] = []
        closest_vals: List[float] = []
        far_vals: List[float] = []

        # Run the same map with different seeds (different goal sets)
        for i in range(n_seeds):
            seed = seed0 + i
            goals = sample_random_goals(env, seed=seed)

            r_pp_default = run_pp(env, goals, priority_mode="default", run_seed=seed)
            r_pp_random  = run_pp(env, goals, priority_mode="random",  run_seed=seed)
            r_pp_closest = run_pp(env, goals, priority_mode="closest", run_seed=seed)
            r_pp_far     = run_pp(env, goals, priority_mode="far",     run_seed=seed)

            for r in (r_pp_default, r_pp_random, r_pp_closest, r_pp_far):
                r.map_name = map_name
                all_rows.append(r)

            m_pp_default = Analyzer.compute_metrics(r_pp_default)
            m_pp_random  = Analyzer.compute_metrics(r_pp_random)
            m_pp_closest = Analyzer.compute_metrics(r_pp_closest)
            m_pp_far     = Analyzer.compute_metrics(r_pp_far)

            # ---- choose the metric you want to average ----
            # (A) Total path length (your current plot)
            default_vals.append(m_pp_default.total_path_len if (m_pp_default.success and m_pp_default.total_path_len is not None) else float("nan"))
            random_vals.append( m_pp_random.total_path_len  if (m_pp_random.success  and m_pp_random.total_path_len  is not None) else float("nan"))
            closest_vals.append(m_pp_closest.total_path_len if (m_pp_closest.success and m_pp_closest.total_path_len is not None) else float("nan"))
            far_vals.append(    m_pp_far.total_path_len     if (m_pp_far.success     and m_pp_far.total_path_len     is not None) else float("nan"))

            # (B) If instead you want runtime, swap to:
            # default_vals.append(m_pp_default.runtime_s if (m_pp_default.success and m_pp_default.runtime_s is not None) else float("nan"))
            # ...

        mu, sd, k = mean_std_ignore_nan(default_vals)
        pp_default_mean.append(mu); pp_default_std.append(sd)

        mu, sd, k = mean_std_ignore_nan(random_vals)
        pp_random_mean.append(mu); pp_random_std.append(sd)

        mu, sd, k = mean_std_ignore_nan(closest_vals)
        pp_closest_mean.append(mu); pp_closest_std.append(sd)

        mu, sd, k = mean_std_ignore_nan(far_vals)
        pp_far_mean.append(mu); pp_far_std.append(sd)

    Analyzer.save_csv(all_rows, f"{out_dir}/{out_csv}")

    Analyzer.plot_pp_with_std_curves(
        x_values,
        pp_default_mean, pp_random_mean, pp_closest_mean, pp_far_mean,
        xlabel=x_label,
        ylabel="Total path length",
        title=f"Total path length vs {x_label} (mean ± std over {n_seeds} seeds)",
        out_path=f"{out_dir}/{out_png}",
        y_pp_std=pp_default_std,
        y_pp_random_std=pp_random_std,
        y_pp_closest_std=pp_closest_std,
        y_pp_far_std=pp_far_std,
    )

    print(f"[OK] sweep(mean±std) -> results/{out_png} and results/{out_csv}")


def main():
    os.makedirs("results", exist_ok=True)

    # 1) Presentation fixed-case
    
    #create_table("maps/map3.txt")  # <-- pick your “harder” map
    #create_table("maps/robots_8_map50_16obs.txt")

    # 2) If you have robot sweep maps like maps/robots_2.txt ... robots_10.txt:
    # # number of robots
    # robot_maps = [
    #     ("maps/robots_2.txt", 2),
    #     ("maps/robots_3.txt", 3),
    #     ("maps/robots_5.txt", 5),
    #     ("maps/robots_7.txt", 7),
    #     ("maps/robots_8.txt", 8),
    #     ("maps/robots_9.txt", 9),
    #     ("maps/robots_10.txt", 10),
    #     ("maps/robots_12.txt", 12),
    #     ("maps/robots_15.txt", 15),
    # ]
    #map size
    robot_maps = [
        ("maps/robots_8_map25.txt", 25),
        ("maps/robots_8_map50.txt", 50),
        ("maps/robots_8_map80.txt", 80),
        ("maps/robots_8_map100.txt", 100),
        ("maps/robots_8_map200.txt", 120),
    ]
    #obstacle
    # robot_maps = [
    #     ("maps/robots_8_map50_8obs.txt", 8),
    #     ("maps/robots_8_map50.txt", 10),
    #     ("maps/robots_8_map50_12obs.txt", 12),
    #     ("maps/robots_8_map50_14obs.txt", 14),
    #     ("maps/robots_8_map50_16obs.txt", 16),
    # ]
    existing = [(p, x) for (p, x) in robot_maps if os.path.exists(p)]
    if len(existing) >= 2:
        paths = [p for p, _ in existing]
        xs = [x for _, x in existing]
        sweep_runtime_by_map(
            map_paths=paths,
            x_label="# of robots",
            x_values=xs,
            out_png="Total path length vs # of robots (mean±std).png",
            out_csv="pp_raw_runs.csv",
            n_seeds=10,
            seed0=300,
        )
    else:
        print("[SKIP] robot sweep maps not found. Generate them (see tools/gen_maps.py).")


if __name__ == "__main__":
    main()
