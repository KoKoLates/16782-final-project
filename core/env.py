import os
import heapq
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple


class Env:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
        
        self.w: int = 0
        self.h: int = 0
        self.robot_num: int = 0
        self.obstacles: List = []

        self._parse_env_from_file(file_path)
        self._build_occupancy_map()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.w, self.h
    
    @property
    def robots_number(self) -> int:
        return self.robot_num
    
    def is_obstacle(self, x: int, y: int) -> bool:
        return self.map[x, y] >= 1e9
    
    def get_cost(self, x: int, y: int) -> float:
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")
        return float(self.map[x, y])
    
    def plot_map(self) -> None:
        arr = self.map.T
        obstacle_mask = arr >= 1e9
        unreachable_mask = (arr >= 1e5) & (~obstacle_mask)
        normal_mask = ~(obstacle_mask | unreachable_mask)

        display = np.zeros_like(arr)

        display[obstacle_mask] = -2
        display[unreachable_mask] = -1

        if np.any(normal_mask):
            norm = arr[normal_mask]
            display[normal_mask] = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

        from matplotlib.colors import ListedColormap, BoundaryNorm
        base = plt.cm.viridis(np.linspace(0, 1, 256))
        cmap = ListedColormap(["black", "gray"] + list(base))

        bounds = [-2.5, -1.5, -0.5] + list(np.linspace(0, 1, 256))
        norm = BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(7, 7))
        plt.imshow(display, cmap=cmap, norm=norm, origin="lower")
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    
    def _parse_env_from_file(self, file: str) -> None:
        with open(file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            if line.startswith("grid_size"):
                self.w, self.h = map(int, line.split(":")[1].split(","))
            
            elif line.startswith("robot_num"):
                self.robot_num = int(line.split(":")[1])

            elif line[0].isdigit():
                x, y, w, h, theta = map(float, line.split(","))
                self.obstacles.append(Env.rotate(x, y, w, h, theta))

    def _build_occupancy_map(self) -> np.ndarray:
        self.map = np.zeros((self.w, self.h), dtype=np.float32) 
        for obs in self.obstacles:
            self._fill(obs)

        self._dijkstra()

    def _fill(self, poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]

        min_x, max_x = max(int(min(xs)), 0), min(int(max(xs)), self.w - 1)
        min_y, max_y = max(int(min(ys)), 0), min(int(max(ys)), self.h - 1)

        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                for dx in (0.0, 0.5, 1.0):
                    for dy in (0.0, 0.5, 1.0):
                        if Env.point_in_polygon(i + dx, j + dy, poly):
                            self.map[i, j] = 1e9
                            break

    def _dijkstra(self):
        w, h = self.w, self.h
        sx, sy = w // 2, h // 2

        if self.is_obstacle(sx, sy):
            raise ValueError("Station is on obstacle.")
        
        dist: np.ndarray = np.full((w, h), np.inf, dtype=np.float32)
        dist[sx, sy] = 0

        pq = [(0.0, sx, sy)]
        move = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while pq:
            cost, x, y = heapq.heappop(pq)
            if cost > dist[x, y]: continue

            for dx, dy in move:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h): continue
                if self.is_obstacle(nx, ny): continue

                nd = cost + 1.0
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    heapq.heappush(pq, (nd, nx, ny))

        dist[~np.isfinite(dist)] = 1e5
        dist[self.map >= 1e9] = 1e9
        self.map = dist

    @staticmethod
    def rotate(x, y, w, h, theta):
        cx, cy = x, y
        theta = np.deg2rad(theta)

        corner = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ])
        R: np.ndarray = np.array([
            [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]
        ])
        return (corner @ R.T + np.array([cx, cy])).tolist()
        
    @staticmethod
    def point_in_polygon(x, y, poly) -> bool:
        inside: bool = False
        length: int = len(poly)

        for i in range(length):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % length]

            if y1 == y2: continue

            condition_1 = (y1 > y) != (y2 > y)
            condition_2 = x < ((x2 - x1) * (y - y1) / (y2 - y1) + x1)

            if condition_1 and condition_2:
                inside = not inside

        return inside
