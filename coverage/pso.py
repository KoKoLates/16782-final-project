import random
import math
import time
import numpy as np
from typing import List, Tuple
from core import Env
from dataclasses import dataclass
from numba import jit

@jit(nopython=True)
def jit_obstacle_block(x1, y1, x2, y2, map_data):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    num_samples = int(dist)
    
    if num_samples == 0: return False
    
    stepx = dx / dist
    stepy = dy / dist
    
    for k in range(1, num_samples):
        px = int(x1 + stepx * k)
        py = int(y1 + stepy * k)
        if px < 0 or py < 0 or px >= map_data.shape[0] or py >= map_data.shape[1]:
            continue
        if map_data[px, py] == 1:
            return True
    return False

@jit(nopython=True)
def jit_add_signal(signal_map, map_data, cx, cy, radius, gain):
    w, h = signal_map.shape
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius + 1))
    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius + 1))
    
    if x_max <= x_min or y_max <= y_min: return

    sigma2 = 2.0 * (radius / 2.0)**2
    
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            dist_sq = (i - cx)**2 + (j - cy)**2
            if dist_sq <= radius**2:
                val = gain * math.exp(-dist_sq / sigma2)
                if jit_obstacle_block(cx, cy, i, j, map_data):
                    val *= 0.5
                signal_map[i, j] += val

@jit(nopython=True)
def jit_calculate_coverage(map_data, positions, connected, tower_pos, 
                           tower_gain, robot_gain, signal_max, signal_threshold):
    w, h = map_data.shape
    signal_map = np.zeros((w, h), dtype=np.float32)
    
    # Tower
    jit_add_signal(signal_map, map_data, tower_pos[0], tower_pos[1], tower_gain, 10.0)
    
    # Robot
    for k in range(len(positions)):
        if connected[k]:
            jit_add_signal(signal_map, map_data, positions[k][0], positions[k][1], robot_gain, 10.0)
            
    # Count Uncovered
    covered_count = 0
    for i in range(w):
        for j in range(h):
            if map_data[i, j] == 1:
                signal_map[i, j] = 0.0 
            else:
                if signal_map[i, j] > signal_max: signal_map[i, j] = signal_max
                if signal_map[i, j] >= signal_threshold: covered_count += 1
                    
    return (w * h) - covered_count

@jit(nopython=True)
def jit_bfs_distance_map(map_data, start_x, start_y):
    w, h = map_data.shape
    dist_map = np.full((w, h), -1, dtype=np.int32)
    queue = np.zeros((w * h, 2), dtype=np.int32)
    head = 0; tail = 0
    if map_data[start_x, start_y] == 0:
        dist_map[start_x, start_y] = 0
        queue[tail, 0] = start_x; queue[tail, 1] = start_y; tail += 1
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)
    while head < tail:
        cx, cy = queue[head]; head += 1
        current_dist = dist_map[cx, cy]
        for i in range(4):
            nx = cx + directions[i, 0]; ny = cy + directions[i, 1]
            if 0 <= nx < w and 0 <= ny < h:
                if map_data[nx, ny] == 0 and dist_map[nx, ny] == -1:
                    dist_map[nx, ny] = current_dist + 1
                    queue[tail, 0] = nx; queue[tail, 1] = ny; tail += 1
    return dist_map

@dataclass
class PSOParams:
    particle_count: int = 75 # 75-100
    max_iter: int = 500
    omega: float = 0.7
    c1: float = 1.4
    c2: float = 1.4
    
    # Signal
    tower_gain: int = 10
    robot_gain: int = 4
    signal_threshold: float = 1.0
    signal_max: float = 10.0
    
    # Weights
    w_obstacle: float = 1e9
    w_hard_constraint: float = 1e7
    w_connect: float = 1e5
    w_coverage: float = 1.0
        
    w_collision: float = 1e4    
    use_collision: bool = False  # repulsion
    collision_dist: float = 2.0 
    
    # Init
    init_method: str = "connected"  # "connected" or "random"
    patience: int = 10
    early_stop: float = 1.0

class Particle:
    def __init__(self, env: Env, params: PSOParams, start_center=None, start_radius=None, map_data=None, reach_map=None):
        self.w, self.h = env.shape
        self.robots_number = env.robots_number
        self.params = params
        self.map_data = map_data
        self.reach_map = reach_map 
        
        # Init Position
        margin = int(params.robot_gain)
        self.safe_x = (margin, self.w - margin - 1)
        self.safe_y = (margin, self.h - margin - 1)
        
        self.position = []
        for _ in range(self.robots_number):
            self.position.append(self._get_valid_position(start_center, start_radius))
            
        # Init Velocity
        self.velocity = [(0.0, 0.0) for _ in range(self.robots_number)]
        
        self.best_position = list(self.position)
        self.best_cost = float("inf")

    def _get_valid_position(self, center=None, radius=None):
        min_x, max_x = self.safe_x
        min_y, max_y = self.safe_y
        
        current_radius = radius
        attempt_count = 0 
        
        while True:
            attempt_count += 1
            
            if attempt_count > 100:
                center = None 
            
            if attempt_count > 2000:
                return (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

            if center and current_radius:
                nx = center[0] + random.uniform(-current_radius, current_radius)
                ny = center[1] + random.uniform(-current_radius, current_radius)
                rx = max(min_x, min(max_x, nx))
                ry = max(min_y, min(max_y, ny))
            else:
                rx = random.uniform(min_x, max_x)
                ry = random.uniform(min_y, max_y)
            
            if self.map_data is not None and self.reach_map is not None:
                ix, iy = int(rx), int(ry)
                if 0 <= ix < self.w and 0 <= iy < self.h:
                    if self.map_data[ix, iy] == 0 and self.reach_map[ix, iy] != -1:
                        return (rx, ry)
            
            if center and current_radius:
                current_radius *= 1.1
class ParticleSwarmOptimizer:
    def __init__(self, env: Env, params: PSOParams = None):
        self.env = env
        self.params = params if params else PSOParams()
        self.w, self.h = env.shape
        self.tower_pos = (float(self.w // 2), float(self.h // 2))

        if hasattr(self.env, 'map'):
            self.map_data = np.array(self.env.map, dtype=np.int32)
        else:
            self.map_data = np.zeros((self.w, self.h), dtype=np.int32)
            for x in range(self.w):
                for y in range(self.h):
                    if self.env.is_obstacle(x, y):
                        self.map_data[x, y] = 1
        
        tx, ty = int(self.tower_pos[0]), int(self.tower_pos[1])
        self.reach_map = jit_bfs_distance_map(self.map_data, tx, ty)

        try:
            threshold = max(1e-3, self.params.signal_threshold)
            term = 20.0 / threshold
            physics_factor = 0.0 if term <= 1.0 else math.sqrt(0.5 * math.log(term))
        except ValueError:
            physics_factor = 1.0
        self.connect_factor = min(1.0, physics_factor)
        
        self.tower_range_sq = ((self.params.tower_gain + self.params.robot_gain) * self.connect_factor) ** 2
        self.robot_range_sq = ((self.params.robot_gain + self.params.robot_gain) * self.connect_factor) ** 2

        is_conn = (self.params.init_method == "connected")
        init_r = (self.params.tower_gain * 2 * self.connect_factor) if is_conn else None
        init_c = self.tower_pos if is_conn else None

        self.particles = [
            Particle(env, self.params, init_c, init_r, self.map_data, self.reach_map)
            for _ in range(self.params.particle_count)
        ]

        self.global_best_position = list(self.particles[0].position)
        self.global_best_cost = float('inf')

    def cobstacle(self, positions):
        cost_obs = 0
        min_x, max_x = self.particles[0].safe_x
        min_y, max_y = self.particles[0].safe_y
        
        for (x, y) in positions:
            ix, iy = int(x), int(y)

            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                cost_obs += self.params.w_obstacle
                continue
            
            if self.map_data[ix, iy] == 1:
                cost_obs += self.params.w_obstacle
                continue

            if self.reach_map[ix, iy] == -1:
                cost_obs += self.params.w_obstacle

        return cost_obs

    def cconnectivity(self, positions):
        n = len(positions)
        connected = [False] * n
        queue = []
        tx, ty = self.tower_pos
        
        # Tower Connect 
        for i, (px, py) in enumerate(positions):
            if (px - tx)**2 + (py - ty)**2 < self.tower_range_sq:
                connected[i] = True
                queue.append(i)

        if not queue:
            dist_pen = sum(math.sqrt((p[0]-tx)**2 + (p[1]-ty)**2) * 100 for p in positions)
            return self.params.w_hard_constraint + dist_pen, connected

        # Robot Connect (BFS)
        head = 0
        while head < len(queue):
            curr = queue[head]; head += 1
            cx, cy = positions[curr]
            for j in range(n):
                if not connected[j]:
                    if (cx - positions[j][0])**2 + (cy - positions[j][1])**2 < self.robot_range_sq:
                        connected[j] = True
                        queue.append(j)
                        
        if all(connected): return 0, connected

        # Disconnected Penalty + Gradient
        cost_connect = 0
        connected_indices = [i for i, c in enumerate(connected) if c]
        disconnected_indices = [i for i, c in enumerate(connected) if not c]
        
        for i in disconnected_indices:
            ix, iy = positions[i]
            min_d = (ix - tx)**2 + (iy - ty)**2
            for j in connected_indices:
                d = (ix - positions[j][0])**2 + (iy - positions[j][1])**2
                if d < min_d: min_d = d
            cost_connect += self.params.w_connect + (math.sqrt(min_d) * 10)

        return cost_connect, connected

    def ccollision(self, positions):
        if not self.params.use_collision:
            return 0.0

        cost = 0
        threshold_sq = self.params.collision_dist ** 2 
        tx, ty = self.tower_pos
        n = len(positions)

        for i in range(n):
            px, py = positions[i]
            
            d_tower_sq = (px - tx)**2 + (py - ty)**2
            if d_tower_sq < threshold_sq:
                cost += self.params.w_collision

            for j in range(i + 1, n):
                qx, qy = positions[j]
                d_robot_sq = (px - qx)**2 + (py - qy)**2
                if d_robot_sq < threshold_sq:
                    cost += self.params.w_collision
        
        return cost
    
    def evaluate(self, positions: List[Tuple[float,float]]) -> float:
        # 1. Obstacle 
        cost_obs = self.cobstacle(positions)
        if cost_obs > 0: return cost_obs + self.params.w_hard_constraint
        
        # 2. Connectivity 
        cost_connect, connected = self.cconnectivity(positions)
        if cost_connect > 0: return cost_obs + cost_connect + self.params.w_hard_constraint

        cost_coll = self.ccollision(positions)
        
        # 3. Coverage
        pos_arr = np.array(positions, dtype=np.float64)
        conn_arr = np.array(connected, dtype=np.bool_)
        cost_cover = jit_calculate_coverage(
            self.map_data, pos_arr, conn_arr, self.tower_pos,
            float(self.params.tower_gain), float(self.params.robot_gain), 
            float(self.params.signal_max), float(self.params.signal_threshold)
        )
        
        return cost_obs + cost_connect + cost_coll + cost_cover * self.params.w_coverage

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

            new_vx = (self.params.omega * vx +
                      self.params.c1 * r1 * (bpx - px) +
                      self.params.c2 * r2 * (gbx - px))
            new_vy = (self.params.omega * vy +
                      self.params.c1 * r1 * (bpy - py) +
                      self.params.c2 * r2 * (gby - py))

            new_x = px + new_vx
            new_y = py + new_vy

            # Clamp to Safe Area
            new_x = max(min_x, min(max_x, new_x))
            new_y = max(min_y, min(max_y, new_y))

            new_positions.append((new_x, new_y))
            new_velocities.append((new_vx, new_vy))

        particle.position = new_positions
        particle.velocity = new_velocities

    def process(self) -> List[Tuple[int, int]]:
        # Initial Evaluate
        for particle in self.particles:
            cost = self.evaluate(particle.position)
            particle.best_cost = cost
            particle.best_position = list(particle.position)
            
            if cost < self.global_best_cost:
                self.global_best_cost = cost
                self.global_best_position = list(particle.position)

        history_global_best_cost = self.global_best_cost
        stall = 0
        start_time = time.time()

        for iteration in range(self.params.max_iter):
            for particle in self.particles:
                self.update_particle(particle)
                
                # Evaluate new position
                cost = self.evaluate(particle.position)

                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = list(particle.position)

                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_position = list(particle.position)

            # Early Stopping
            delta_cost = abs(history_global_best_cost - self.global_best_cost)
            if delta_cost < self.params.early_stop:
                stall += 1
            else:
                stall = 0

            if stall >= self.params.patience:
                print(f"Early stop at Iter {iteration+1}")
                break

            history_global_best_cost = self.global_best_cost
        
        end_time = time.time()
        print(f"PSO End. Time: {end_time - start_time:.4f}s, Best Cost: {self.global_best_cost:.2f}")
        
        result = [(int(x), int(y)) for (x, y) in self.global_best_position]
        return result