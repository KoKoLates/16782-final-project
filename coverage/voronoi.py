import numpy as np

from core import Env


class VoronoiOptimizer:
    def __init__(self, env: Env):
        self.num_robots = env.robots_number
        self.w, self.h = env.shape
        self.env = env

        free = []
        for x in range(self.w):
            for y in range(self.h):
                if not env.is_obstacle(x, y):
                    free.append((x, y))

        self.free_points = np.array(free, dtype=float)

        if len(self.free_points) < self.num_robots:
            raise ValueError("Not enough free space for given robots.")

        idx = np.random.choice(len(self.free_points), size=self.num_robots, replace=False)
        self.robot_positions = self.free_points[idx].copy()

    def _compute_centroid(self, points):
        if len(points) == 0:
            return None
        return np.mean(points, axis=0)

    def process(self, iterations: int = 10):
        for _ in range(iterations):
            diff = self.free_points[:, None, :] - self.robot_positions[None, :, :]
            dist = np.sum(diff * diff, axis=2)
            nearest_robot = np.argmin(dist, axis=1)

            new_positions = []
            for r in range(self.num_robots):
                pts = self.free_points[nearest_robot == r]
                centroid = self._compute_centroid(pts)
                if centroid is None:
                    centroid = self.robot_positions[r]
                new_positions.append(centroid)

            self.robot_positions = np.array(new_positions)

        return [(float(x), float(y)) for x, y in self.robot_positions]
