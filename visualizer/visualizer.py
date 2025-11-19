import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.patches import Circle

from core import Env
from planner import Path
from typing import List, Tuple


class Visualizer:
    def __init__(self, env: Env):
        self.env = env
        self.w, self.h = env.shape

    def _prepare_ax(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # obstacles
        for obs in self.env.obstacles:
            poly = Polygon(obs, closed=True, color="black", alpha=0.7)
            ax.add_patch(poly)

        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect("equal")
        ax.grid(True, color="gray", linewidth=0.5, alpha=0.3)

        return fig, ax

    def plot_points(self, points: List[Tuple[int, int]]):
        fig, ax = self._prepare_ax()

        cmap = cm.get_cmap("tab20", len(points))

        for i, p in enumerate(points):
            x, y = p
            ax.scatter(x, y, color=cmap(i), s=60, label=f"P{i}")

        ax.legend()
        plt.show()

    def plot_paths(self, paths: List[Path]):
        fig, ax = self._prepare_ax()

        cmap = cm.get_cmap("tab20", len(paths))

        for i, path in enumerate(paths):
            xs = [s.x for s in path]
            ys = [s.y for s in path]
            ax.plot(xs, ys, color=cmap(i), linewidth=2, label=f"Path {i}")

        ax.legend()
        plt.show()

    def animate(self, paths: List[Path], interval: int = 200, file_name: str = "./cache/assets/animation.gif"):
        fig, ax = self._prepare_ax()

        cmap = cm.get_cmap("tab20", len(paths))
        scatters = []
        traces = []
        goal_circles = []

        goals = [path[-1] for path in paths]

        for goal in goals:
            circle = Circle(
                (goal.x, goal.y),
                radius=0.1,     
                facecolor="none",
                edgecolor="gray",
                linewidth=1,
                zorder=5
            )
            ax.add_patch(circle)
            goal_circles.append(circle)

        for i, path in enumerate(paths):
            sc = ax.scatter([], [], color=cmap(i), s=20, zorder=6)
            scatters.append(sc)

            tr, = ax.plot([], [], color=cmap(i), linewidth=2, alpha=0.7)
            traces.append(tr)

        all_times = sorted({s.t for path in paths for s in path})
        t_min, t_max = all_times[0], all_times[-1]

        indexed = [{s.t: (s.x, s.y) for s in path} for path in paths]

        def update(frame_t):
            for i, d in enumerate(indexed):
                valid_ts = [t for t in d.keys() if t <= frame_t]
                if not valid_ts:
                    continue

                t = max(valid_ts)
                x, y = d[t]

                # Update live robot marker
                scatters[i].set_offsets([[x, y]])

                # Update trajectory
                past = [d[tt] for tt in sorted(valid_ts)]
                xs = [p[0] for p in past]
                ys = [p[1] for p in past]
                traces[i].set_data(xs, ys)

                # Update goal circle edge when reached
                if frame_t >= goals[i].t:
                    goal_circles[i].set_edgecolor(cmap(i))
                else:
                    goal_circles[i].set_edgecolor("gray")  # still no fill

            return scatters + traces + goal_circles

        ani = FuncAnimation(
            fig,
            update,
            frames=range(t_min, t_max + 1),
            interval=interval,
            blit=True,
            repeat=False
        )
        ani.save(file_name, writer="pillow", fps=1000/interval)

        plt.show()
