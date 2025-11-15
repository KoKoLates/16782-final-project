import os
import numpy as np


class Env:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
        
        self.num = 0
        self.w, self.h = 0, 0
        self.obstacles = []

        self._load(file_path)

        self.map = np.zeros((self.w, self.h), dtype=np.int8)
        for poly in self.obstacles:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]

            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))
            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, self.w - 1)
            max_y = min(max_y, self.h - 1)

            for i in range(min_x, max_x + 1):
                for j in range(min_y, max_y + 1):
                    if Env.inside(i + 0.5, j + 0.5, poly):
                        self.map[i, j] = 1

    def is_obstacle(self, x, y):
        return self.map[x, y]
    
    @property
    def shape(self):
        return self.w, self.h
    
    @property
    def robots_number(self):
        return self.num

    def _load(self, file: str):
        with open(file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            if line.startswith("grid_size"):
                v = line.split(":")[1].strip()
                self.w, self.h = map(int, v.split(","))

            elif line.startswith("robot_num"):
                v = line.split(":")[1].strip()
                self.num = int(v)

            elif line.startswith("obstacles"):
                continue

            elif line[0].isdigit():
                x, y, w, h, theta = list(map(float, line.split(",")))
                self.obstacles.append(Env.rotate(x, y, w, h, theta))

    @staticmethod
    def rotate(x, y, w, h, theta):
        cx, cy = x + w / 2, y + h / 2
        theta = np.deg2rad(theta)

        dx, dy = w / 2, h / 2
        corner = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]

        rect = []
        for px, py in corner:
            rx = cx + px * np.cos(theta) - py * np.sin(theta)
            ry = cy + px * np.sin(theta) + py * np.cos(theta)
            rect.append((rx, ry))

        return rect
    
    @staticmethod
    def inside(x, y, poly):
        count: bool = False
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]

            if y1 == y2: continue

            if ((y1 > y != y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                count = not count
        
        return count