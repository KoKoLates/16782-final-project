import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection

from core import Env
from planner import Path
from typing import List, Tuple, Optional


class Visualizer:
    def __init__(self, env: Env):
        self.env = env
        self.w, self.h = env.shape

    def plot_points(self, points: List[Tuple[int, int]]) -> None:
        _, ax = self._prepare_ax()
        cmap = cm.get_cmap("tab20", len(points))

        ax.scatter(self.w / 2, self.h / 2, marker='x', color='gray', s=80, linewidths=2)
        center_circle = Circle((self.w / 2, self.h / 2), 10, edgecolor='gray', fill=False, linewidth=1.5)
        ax.add_patch(center_circle)


        for i, p in enumerate(points):
            x, y = p
            ax.scatter(x, y, color=cmap(i), s=60, label=f'robot {i + 1}')

            circle = Circle((x, y), radius=4, edgecolor=cmap(i), fill=False, linewidth=1.5)
            ax.add_patch(circle)

        # ax.legend()
        plt.show()

    # def plot_points(self, points: List[Tuple[int, int]]) -> None:
    #     fig, ax = self._prepare_ax()
    #     cmap = cm.get_cmap("tab20", len(points))
    #     center = (self.w / 2, self.h / 2)

    #     ax.scatter(*center, marker='x', color='gray', s=80, linewidths=2)
    #     ax.add_patch(Circle(center, 10, edgecolor='gray', fill=False, linewidth=1.5))

    #     for i, (x, y) in enumerate(points):
    #         ax.scatter(x, y, color=cmap(i), s=60)
    #         ax.add_patch(Circle((x, y), radius=4, edgecolor=cmap(i), fill=False, linewidth=1.5))

        # nodes, edges = self._compute_connectivity(points)
        # for i, j in edges:
        #     x1, y1 = nodes[i]
        #     x2, y2 = nodes[j]
        #     ax.plot([x1, x2], [y1, y2], color='black', linewidth=1, alpha=0.7)

        # plt.show()

    def plot_paths(self, paths: List[Path]):
        _, ax = self._prepare_ax()

        cmap = cm.get_cmap("tab20", len(paths))

        for i, path in enumerate(paths):
            xs = [s.x for s in path]
            ys = [s.y for s in path]
            ax.plot(xs, ys, color=cmap(i), linewidth=2)

        plt.show()

    def animate(self, paths: List[Path], file_name: Optional[str] = None, interval: int = 200):
        fig, ax = self._prepare_ax()

        cmap = cm.get_cmap("tab20", len(paths))

        scatters: List[PathCollection] = []
        traces: List[Line2D] = []
        goal_circles: List[Circle] = []

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

        for i, _ in enumerate(paths):
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

                scatters[i].set_offsets([[x, y]])
                past = [d[tt] for tt in sorted(valid_ts)]
                xs = [p[0] for p in past]
                ys = [p[1] for p in past]
                traces[i].set_data(xs, ys)

                if frame_t >= goals[i].t:
                    goal_circles[i].set_edgecolor(cmap(i))
                else:
                    goal_circles[i].set_edgecolor("gray")

            return scatters + traces + goal_circles

        ani = FuncAnimation(
            fig,
            update,
            frames=range(t_min, t_max + 1),
            interval=interval,
            blit=True,
            repeat=False
        )
        if file_name is not None:
            ani.save(file_name, writer="pillow", fps=1000/interval)

        plt.show()

    def _prepare_ax(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        for obs in self.env.obstacles:
            poly = Polygon(obs, closed=True, color="black", alpha=0.7)
            ax.add_patch(poly)

        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect("equal")
        ax.grid(True, color="gray", linewidth=0.5, alpha=0.3)

        return fig, ax
    
    def _compute_connectivity(self, points: List[Tuple[int, int]]):
        center = (self.w / 2, self.h / 2)
        nodes = [center] + points
        N = len(nodes)

        graph = [[] for _ in range(N)]

        for i in range(1, N):
            if np.hypot(nodes[i][0] - center[0], nodes[i][1] - center[1]) < 14:
                graph[0].append(i)
                graph[i].append(0)


        for i in range(1, N):
            for j in range(i + 1, N):
                if np.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1]) < 8:
                    graph[i].append(j)
                    graph[j].append(i)

        visited = set()
        queue = [0]
        while queue:
            u = queue.pop(0)
            if u in visited:
                continue
            visited.add(u)
            for v in graph[u]:
                if v not in visited:
                    queue.append(v)

        edges = []
        for i in visited:
            for j in graph[i]:
                if j in visited and i < j:
                    edges.append((i, j))

        return nodes, edges
