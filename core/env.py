import os
import numpy as np

from typing import List, Tuple
from matplotlib.path import Path


class Env:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
        
        self.w, self.h = 0, 0
        self.robot_num = 0
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
        return bool(self.map[x, y])
    
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
        self.map = np.zeros((self.w, self.h), dtype=np.uint8) 
        for obs in self.obstacles:
            self._fill(obs)

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
                            self.map[i, j] = 1

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
