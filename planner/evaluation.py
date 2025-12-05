from __future__ import annotations

import os
import csv
import time
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Sequence
import matplotlib.pyplot as plt
from .node import State, Path
import os
from typing import Optional
import pandas as pd
import re

def create_animation(env, paths, file_name: str, interval: int = 200, enabled: bool = True) -> Optional[str]:
    """
    Saves a GIF animation if enabled and paths is non-empty.
    Returns file_name if saved, else None.
    """
    if not enabled:
        return None
    if not paths:
        return None

    # lazy import to avoid matplotlib animation overhead when not needed
    from visualizer.visualizer import Visualizer  # adjust if your module path differs

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    vis = Visualizer(env)
    vis.animate(paths, interval=interval, file_name=file_name)
    return file_name


Pos = Tuple[int, int]

@dataclass
class PlannerRun:
    planner: str
    map_name: str
    n_robots: int
    n_obstacles: Optional[int]
    success: bool
    runtime_s: float
    paths: List[Path]  # List[List[State]]


@dataclass
class MetricsRow:
    planner: str
    map_name: str
    n_robots: int
    n_obstacles: Optional[int]
    success: bool

    runtime_s: Optional[float]

    total_path_len: Optional[float]  # SOC
    avg_path_len: Optional[float]
    std_path_len: Optional[float]
    makespan: Optional[float]


# analyzer class
class Analyzer:
    @staticmethod
    def ensure_dir(d: str) -> None:
        os.makedirs(d, exist_ok=True)

    @staticmethod
    def extract_xy(s: State) -> Pos:
        return int(s.x), int(s.y)

    @staticmethod
    def path_cost(path: Path) -> int:
        # path includes start at t=0 => steps = len-1 (waits count because states are time-indexed)
        return max(0, len(path) - 1)

    @staticmethod
    def _pad_paths(pos_paths: Dict[int, List[Pos]]) -> Dict[int, List[Pos]]:
        if not pos_paths:
            return pos_paths
        L = max(len(p) for p in pos_paths.values())
        out: Dict[int, List[Pos]] = {}
        for i, p in pos_paths.items():
            if len(p) < L:
                p = p + [p[-1]] * (L - len(p))
            out[i] = p
        return out


    @staticmethod
    def compute_metrics(run: PlannerRun) -> MetricsRow:
        if (not run.success) or (not run.paths):
            return MetricsRow(
                planner=run.planner,
                map_name=run.map_name,
                n_robots=run.n_robots,
                n_obstacles=run.n_obstacles,
                success=False,
                runtime_s=run.runtime_s,
                total_path_len=None,
                avg_path_len=None,
                std_path_len=None,
                makespan=None,
            )

        lens = [Analyzer.path_cost(p) for p in run.paths]
        total_path_len = float(sum(lens))
        avg_path_len = float(statistics.mean(lens))
        std_path_len = float(statistics.pstdev(lens)) if len(lens) > 1 else 0.0
        makespan = float(max(lens)) if lens else 0.0

        return MetricsRow(
            planner=run.planner,
            map_name=run.map_name,
            n_robots=run.n_robots,
            n_obstacles=run.n_obstacles,
            success=True,
            runtime_s=run.runtime_s,
            total_path_len=total_path_len,
            avg_path_len=avg_path_len,
            std_path_len=std_path_len,
            makespan=makespan,
        )

    @staticmethod
    def save_csv(rows: List[MetricsRow], out_path: str) -> None:
        Analyzer.ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    @staticmethod
    def _sanitize_sheet_name(name: str) -> str:
        name = re.sub(r"[:\\/?*\[\]]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return (name[:31] if name else "Sheet1")

    @staticmethod
    def save_xlsx(
        rows: List["MetricsRow"],
        out_path: str,
        sheet_name: str = "results",
        also_save_transposed: bool = True,
        transpose_sheet_name: str = "transposed",
        transpose_index_cols: Sequence[str] = ("planner",),
        transpose_keep_cols: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Writes an .xlsx with:
          - sheet_name: the original row-wise table
          - transpose_sheet_name: a transposed/summary table where planners are columns

        transpose_keep_cols: which metrics to include in the transposed sheet.
          If None, uses a sensible default.
        """
        Analyzer.ensure_dir(os.path.dirname(out_path))

        df = pd.DataFrame([asdict(r) for r in rows])

        # default ordering of columns
        front = ["planner", "success", "runtime_s", "total_path_len", "makespan", "avg_path_len", "std_path_len",
                 "map_name", "n_robots", "n_obstacles"]
        cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
        df = df[cols]

        sheet_name = Analyzer._sanitize_sheet_name(sheet_name)
        transpose_sheet_name = Analyzer._sanitize_sheet_name(transpose_sheet_name)

        if transpose_keep_cols is None:
            transpose_keep_cols = [c for c in ["success", "runtime_s", "total_path_len", "makespan",
                                               "avg_path_len", "std_path_len"] if c in df.columns]

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            # Sheet 1: original
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # Sheet 2: transposed
            if also_save_transposed:
                # build a small table: rows=metrics, cols=planners
                # if multiple runs per planner exist, this will error; in that case aggregate first.
                idx_cols = list(transpose_index_cols)  # usually just ["planner"]
                tdf = df[idx_cols + list(transpose_keep_cols)].copy()

                # Make a single column key for columns (planner)
                if idx_cols != ["planner"]:
                    tdf["planner_key"] = tdf[idx_cols].astype(str).agg(" | ".join, axis=1)
                else:
                    tdf["planner_key"] = tdf["planner"]

                # pivot to: rows=metrics, cols=planner_key
                wide = tdf.set_index("planner_key")[list(transpose_keep_cols)].T
                wide.insert(0, "metric", wide.index)  # first column = metric names
                wide.reset_index(drop=True, inplace=True)

                wide.to_excel(writer, index=False, sheet_name=transpose_sheet_name)
                ws2 = writer.book[transpose_sheet_name]
                ws2.freeze_panes = "B2"   # keep metric column visible
                ws2.auto_filter.ref = ws2.dimensions

    # Plots
    @staticmethod
    def plot_two_curves(
        x: List[Any],
        y_pp: List[float],
        y_cbs: List[float],
        xlabel: str,
        ylabel: str,
        title: str,
        out_path: str,
    ) -> None:
        Analyzer.ensure_dir(os.path.dirname(out_path))
        plt.figure()
        plt.plot(x, y_pp, marker="o", label="Prioritized far")
        plt.plot(x, y_cbs, marker="o", linestyle= "--", label="CBS")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
    
    @staticmethod
    def plot_three_curves(
        x: List[Any],
        y_pp: List[float],
        y_pp_1: List[float],
        y_cbs: List[float],
        xlabel: str,
        ylabel: str,
        title: str,
        out_path: str,
    ) -> None:
        Analyzer.ensure_dir(os.path.dirname(out_path))
        plt.figure()
        plt.plot(x, y_pp, marker="o", label="Prioritized far")
        plt.plot(x, y_pp_1, marker="o", label="Prioritized closest")
        plt.plot(x, y_cbs, marker="o", label="CBS")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        
    
    @staticmethod
    def ensure_dir(path: str) -> None:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def plot_pp_with_std_curves(
        x: List[Any],
        y_pp: List[float],
        y_pp_random: List[float],
        y_pp_closest: List[float],
        y_pp_far: List[float],
        xlabel: str,
        ylabel: str,
        title: str,
        out_path: str,
        # standard deviations for error bars
        y_pp_std: Optional[List[float]] = None,
        y_pp_random_std: Optional[List[float]] = None,
        y_pp_closest_std: Optional[List[float]] = None,
        y_pp_far_std: Optional[List[float]] = None,
    ) -> None:
        Analyzer.ensure_dir(os.path.dirname(out_path))
        plt.figure()

        def _plot_or_errorbar(y, ystd, label):
            if ystd is None:
                plt.plot(x, y, marker="o", label=label)
            else:
                # error bars: mean ± std
                plt.errorbar(x, y, yerr=ystd, marker="o", capsize=3, label=label)

        _plot_or_errorbar(y_pp,         y_pp_std,         "Prioritized default")
        _plot_or_errorbar(y_pp_random,  y_pp_random_std,  "Prioritized random")
        _plot_or_errorbar(y_pp_closest, y_pp_closest_std, "Prioritized closest")
        _plot_or_errorbar(y_pp_far,     y_pp_far_std,     "Prioritized far")
        
        # _plot_or_errorbar(y_pp,         None,         "Prioritized default")
        # _plot_or_errorbar(y_pp_random,  None,  "Prioritized random")
        # _plot_or_errorbar(y_pp_closest, None, "Prioritized closest")
        # _plot_or_errorbar(y_pp_far,     None,     "Prioritized far")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
         
    @staticmethod
    def aggregate_mean_by_x(
        runs: List[MetricsRow],
        x_getter,
        planner_name: str,
        y_getter,
        xs: List[Any],
    ) -> List[float]:
        out = []
        for x in xs:
            samples = [
                y_getter(r)
                for r in runs
                if r.planner == planner_name and r.success and x_getter(r) == x and y_getter(r) is not None
            ]
            out.append(statistics.mean(samples) if samples else float("nan"))
        return out
