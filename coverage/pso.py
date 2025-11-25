import random
import math
import time
import numpy as np
from typing import List, Tuple
from core import Env
from .base import CoverageOptimizer
from .node import Particle

class ParticleSwarmOptimizer(CoverageOptimizer):
    def __init__(
        self, 
        env: Env, 
        particle_count: int = 50,
        max_iter: int = 100,
        omega: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4,
        tower_signal_range: int = 10,
        robot_signal_range: int = 4,
        cost_coverage_threshold: float = 1.0,
        cost_coverage_max_signal: float = 10.0
    ):
        super().__init__(env)
        self.particle_count = particle_count
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.tower_signal_range = tower_signal_range
        self.robot_signal_range = robot_signal_range
        self.cost_coverage_threshold = cost_coverage_threshold
        self.cost_coverage_max_signal = cost_coverage_max_signal

        self.particles = [
            Particle(env)
            for _ in range(particle_count)
        ]

        self.global_best_position = [None] * self.env.robots_number
        self.global_best_cost = float('inf')
    
    def cost_obstacle(self, positions):
        cost_obs = 0
        for (x, y) in positions:
            if self.env.is_obstacle(int(x), int(y)):
                cost_obs += 1e9
        return cost_obs
    
    def cost_connectivity(self, positions):
        tower = (self.env.w / 2, self.env.h / 2)
        tower_connect_range = self.tower_signal_range + self.robot_signal_range
        robot_connect_range = self.robot_signal_range + self.robot_signal_range

        n = len(positions)
        connected = [False] * n

        roots = []
        for i, p in enumerate(positions):
            if math.dist(p, tower) < tower_connect_range:
                roots.append(i)
                connected[i] = True

        if not roots:
            return 1e6, connected

        queue = roots[:]
        while queue:
            i = queue.pop(0)
            for j in range(n):
                if not connected[j] and math.dist(positions[i], positions[j]) < robot_connect_range:
                    connected[j] = True
                    queue.append(j)

        cost_connect = 0
        for ok in connected:
            if not ok:
                cost_connect += 1e5

        return cost_connect, connected
    
    def ray_blocked(self, x1, y1, x2, y2):
        x1, y1 = float(x1), float(y1)
        x2, y2 = float(x2), float(y2)

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)

        num_samples = int(dist)

        if num_samples == 0:
            return False

        stepx = dx / dist
        stepy = dy / dist

        for k in range(1, num_samples):
            px = x1 + stepx * k
            py = y1 + stepy * k

            if self.env.is_obstacle(int(px), int(py)):
                return True

        return False

    def cost_coverage(self, positions, connected):
        w, h = self.env.w, self.env.h
        sigma = self.robot_signal_range / 2
        sigmat = self.tower_signal_range / 2
        tower_x, tower_y = self.env.w / 2, self.env.h / 2

        xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')
        signal_map = np.zeros((w, h), dtype=np.float32)

        # Tower
        dxt = xs - tower_x
        dyt = ys - tower_y
        dt = np.sqrt(dxt*dxt + dyt*dyt)
        withint = (dt <= self.tower_signal_range)
        st = 10 * np.exp(-(dt*dt) / (2 * sigmat * sigmat))
        st *= withint

        within_xt, within_yt = np.where(st > 0)
        for x, y in zip(within_xt, within_yt):
            if self.ray_blocked(tower_x, tower_y, x, y):
                st[x, y] *= 0.5
        
        signal_map += st
        
        # Robot
        for i, (rx, ry) in enumerate(positions):
            if not connected[i]:
                continue
            
            dx = xs - rx
            dy = ys - ry
            d = np.sqrt(dx*dx + dy*dy)

            within = (d <= self.robot_signal_range)

            s = 10 * np.exp(-(d*d) / (2 * sigma * sigma))
            s *= within

            within_x, within_y = np.where(s > 0)
            for x, y in zip(within_x, within_y):
                if self.ray_blocked(rx, ry, x, y):
                    s[x, y] *= 0.5

            signal_map += s

        signal_map = np.minimum(signal_map, self.cost_coverage_max_signal)

        obstacle_mask = (self.env.map == 1)
        signal_map[obstacle_mask] = 0

        covered = (signal_map >= self.cost_coverage_threshold)
        max_cells = w * h
        return max_cells - covered.sum()

    def evaluate(self, positions: List[Tuple[float,float]]) -> float:
        cost_obs = self.cost_obstacle(positions)
        cost_connect, connected = self.cost_connectivity(positions)
        cost_cover = self.cost_coverage(positions, connected)
        return cost_obs + cost_connect + cost_cover

    def update_particle(self, particle: Particle):
        new_positions = []
        new_velocities = []

        for i in range(particle.robots_number):

            px, py = particle.position[i]
            vx, vy = particle.velocity[i]

            bpx, bpy = particle.best_position[i]
            gbx, gby = self.global_best_position[i]

            r1, r2 = random.random(), random.random()

            new_vx = (
                self.omega * vx +
                self.c1 * r1 * (bpx - px) +
                self.c2 * r2 * (gbx - px)
            )

            new_vy = (
                self.omega * vy +
                self.c1 * r1 * (bpy - py) +
                self.c2 * r2 * (gby - py)
            )

            new_x = px + new_vx
            new_y = py + new_vy

            new_x = max(0, min(self.env.w - 1, new_x))
            new_y = max(0, min(self.env.h - 1, new_y))

            new_positions.append((new_x, new_y))
            new_velocities.append((new_vx, new_vy))

        particle.position = new_positions
        particle.velocity = new_velocities
    
    def process(self) -> List[Tuple[int, int]]:
        self.global_best_position = list(self.particles[0].position)
        self.global_best_cost = self.evaluate(self.global_best_position)
        history_global_best_cost = self.global_best_cost
        stall = 0
        epsilon = 1
        patience = 10

        start_time = time.time()

        for iteration in range(self.max_iter):
            for particle in self.particles:
                cost = self.evaluate(particle.position)

                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = list(particle.position)

                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_position = list(particle.position)

            for particle in self.particles:
                self.update_particle(particle)

            delta_cost = abs(history_global_best_cost - self.global_best_cost)

            if delta_cost < epsilon:
                stall += 1
            else:
                stall = 0

            if stall >= patience:
                break

            history_global_best_cost = self.global_best_cost
        
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.4f} seconds")
        print("Iteration Count:", iteration)
        print("Final Best Cost:", self.global_best_cost)
        result = [(int(x), int(y)) for (x, y) in self.global_best_position]
        return result
        