import random
import math
import time
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from core import Env

from .base import Coverage, jit_calculate_coverage, calculate_physics_factor
from .node import Particle, Metric

@dataclass
class PSOParams:
    particle_count: int = 75
    max_iter: int = 100
    omega: float = 0.7
    c1: float = 1.4
    c2: float = 1.4
    
    tower_gain: int = 10
    robot_gain: int = 4
    signal_threshold: float = 1.0
    signal_max: float = 10.0
    
    w_obstacle: float = 1e9
    w_hard_constraint: float = 1e7
    w_connect: float = 1e5
    w_coverage: float = 1.0
    
    w_collision: float = 1e4    
    use_collision: bool = True     # repulsion
    collision_dist: float = 2.0 
    
    init_method: str = "connected" # "connected" or "random"
    patience: int = 20
    early_stop: float = 1.0

class ParticleSwarmOptimizer(Coverage):
    def __init__(self, env: Env, params: PSOParams = None):
        super().__init__(env)
        self.params = params if params else PSOParams()
        self.metric = Metric()

        # Physics
        p_factor = calculate_physics_factor(self.params.signal_threshold)
        self.connect_factor = min(1.0, p_factor)
        
        self.tower_range_sq = ((self.params.tower_gain + self.params.robot_gain) * self.connect_factor) ** 2
        self.robot_range_sq = ((self.params.robot_gain + self.params.robot_gain) * self.connect_factor) ** 2

        # Setup Particles
        margin = int(self.params.robot_gain)
        safe_x = (margin, self.w - margin - 1)
        safe_y = (margin, self.h - margin - 1)

        is_conn = (self.params.init_method == "connected")
        init_r = (self.params.tower_gain * 2 * self.connect_factor) if is_conn else None
        init_c = self.tower_pos if is_conn else None

        self.particles = [
            Particle(self.num_robots, self.map_data, safe_x, safe_y, init_c, init_r)
            for _ in range(self.params.particle_count)
        ]

        self.global_best_position = list(self.particles[0].position)
        self.global_best_cost = float('inf')

    def cobstacle(self, positions):
        cost_obs = 0
        min_x, max_x = self.particles[0].safe_x
        min_y, max_y = self.particles[0].safe_y
        
        for (x, y) in positions:
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                cost_obs += self.params.w_obstacle
                continue
            
            # Check if >= 1e5 (Unreachable or Obstacle)
            val = self.map_data[int(x), int(y)]
            if val >= 1e5:
                cost_obs += self.params.w_obstacle

        return cost_obs

    def cconnectivity(self, positions):
        n = len(positions)
        connected = [False] * n
        queue = []
        
        tx, ty = int(round(self.tower_pos[0])), int(round(self.tower_pos[1]))
        int_positions = [(int(round(x)), int(round(y))) for x, y in positions]

        for i, (ix, iy) in enumerate(int_positions):
            if (ix - tx)**2 + (iy - ty)**2 <= self.tower_range_sq:
                connected[i] = True
                queue.append(i)
                
        if not queue:
            dist_pen = sum(math.sqrt((p[0]-self.tower_pos[0])**2 + (p[1]-self.tower_pos[1])**2) * 100 for p in positions)
            return self.params.w_hard_constraint + dist_pen, connected

        head = 0
        while head < len(queue):
            curr = queue[head]; head += 1
            cx, cy = int_positions[curr] 
            
            for j in range(n):
                if not connected[j]:
                    jx, jy = int_positions[j] 
                    
                    if (cx - jx)**2 + (cy - jy)**2 <= self.robot_range_sq:
                        connected[j] = True
                        queue.append(j)

        if all(connected): return 0, connected
        
        cost_connect = 0
        connected_indices = [i for i, c in enumerate(connected) if c]
        disconnected_indices = [i for i, c in enumerate(connected) if not c]
        
        tower_x_float, tower_y_float = self.tower_pos
        
        for i in disconnected_indices:
            ix, iy = positions[i] 
            min_d = (ix - tower_x_float)**2 + (iy - tower_y_float)**2
            
            for j in connected_indices:
                d = (ix - positions[j][0])**2 + (iy - positions[j][1])**2
                if d < min_d: min_d = d
            
            cost_connect += self.params.w_connect + (math.sqrt(min_d) * 10)
            
        return cost_connect, connected

    def ccollision(self, positions):
        if not self.params.use_collision: return 0.0
        cost = 0
        threshold_sq = self.params.collision_dist ** 2 
        tx, ty = self.tower_pos
        n = len(positions)
        for i in range(n):
            px, py = positions[i]
            if (px - tx)**2 + (py - ty)**2 < threshold_sq: cost += self.params.w_collision
            for j in range(i + 1, n):
                qx, qy = positions[j]
                if (px - qx)**2 + (py - qy)**2 < threshold_sq: cost += self.params.w_collision
        return cost
    
    def evaluate(self, positions: List[Tuple[float,float]]) -> float:
        cost_obs = self.cobstacle(positions)
        if cost_obs > 0: return cost_obs + self.params.w_hard_constraint
        cost_connect, connected = self.cconnectivity(positions)
        if cost_connect > 0: return cost_obs + cost_connect + self.params.w_hard_constraint
        cost_coll = self.ccollision(positions)
        
        pos_arr = np.array(positions, dtype=np.float64)
        conn_arr = np.array(connected, dtype=np.bool_)
        cost_cover = jit_calculate_coverage(
            self.map_data, pos_arr, conn_arr, self.tower_pos,
            float(self.params.tower_gain), float(self.params.robot_gain), 
            float(self.params.signal_max), float(self.params.signal_threshold)
        )

        total = cost_obs + cost_connect + cost_coll + cost_cover * self.params.w_coverage
        self.metric.cost_obs = cost_obs
        self.metric.cost_connect = cost_connect
        self.metric.cost_coll = cost_coll
        self.metric.cost_cover = cost_cover
        self.metric.cost_all = total

        return total

    def compute_metric(self):
        return self.metric.as_tuple()

    def update_particle(self, particle: Particle):
        new_positions = []
        new_velocities = []
        min_x, max_x = particle.safe_x
        min_y, max_y = particle.safe_y

        for i in range(particle.robots_number):
            px, py = particle.position[i]
            vx, vy = particle.velocity[i]
            bpx, bpy = particle.best_position[i]
            gbx, gby = self.global_best_position[i]
            r1, r2 = random.random(), random.random()
            
            new_vx = (self.params.omega * vx + self.params.c1 * r1 * (bpx - px) + self.params.c2 * r2 * (gbx - px))
            new_vy = (self.params.omega * vy + self.params.c1 * r1 * (bpy - py) + self.params.c2 * r2 * (gby - py))
            
            new_x = max(min_x, min(max_x, px + new_vx))
            new_y = max(min_y, min(max_y, py + new_vy))
            
            new_positions.append((new_x, new_y))
            new_velocities.append((new_vx, new_vy))

        particle.position = new_positions
        particle.velocity = new_velocities

    def process(self) -> List[Tuple[int, int]]:
        self.stopped_iter = 0
        for particle in self.particles:
            cost = self.evaluate(particle.position)
            particle.best_cost = cost
            particle.best_position = list(particle.position)
            if cost < self.global_best_cost:
                self.global_best_cost = cost
                self.global_best_position = list(particle.position)

        stall = 0; best_hist = self.global_best_cost

        for iteration in range(self.params.max_iter):
            for particle in self.particles:
                self.update_particle(particle)
                cost = self.evaluate(particle.position)
                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = list(particle.position)
                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_position = list(particle.position)

            if abs(best_hist - self.global_best_cost) < self.params.early_stop: stall += 1
            else: stall = 0; best_hist = self.global_best_cost

            if stall >= self.params.patience:
                self.stopped_iter = iteration + 1
                print(f"Early stop at Iter {iteration+1}")
                print(f"Early stop at Iter {self.stopped_iter}")
                break
        cost = self.evaluate(self.global_best_position)
        print(f"PSO End. Best Cost: {self.global_best_cost:.2f}")
        return [(int(round(x)), int(round(y))) for (x, y) in self.global_best_position]