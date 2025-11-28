import numpy as np
import random
import math
from core import Env
from typing import List, Tuple
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
    
    head = 0
    tail = 0
    
    # Init start
    if map_data[start_x, start_y] == 0:
        dist_map[start_x, start_y] = 0
        queue[tail, 0] = start_x
        queue[tail, 1] = start_y
        tail += 1
    
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)
    
    while head < tail:
        cx, cy = queue[head]
        head += 1
        current_dist = dist_map[cx, cy]
        
        for i in range(4):
            nx = cx + directions[i, 0]
            ny = cy + directions[i, 1]
            
            if 0 <= nx < w and 0 <= ny < h:
                if map_data[nx, ny] == 0 and dist_map[nx, ny] == -1:
                    dist_map[nx, ny] = current_dist + 1
                    queue[tail, 0] = nx
                    queue[tail, 1] = ny
                    tail += 1
                    
    return dist_map

@jit(nopython=True)
def jit_calculate_travel_cost(positions, dist_maps):
    total_dist = 0.0
    penalty = 0.0
    
    for i in range(len(positions)):
        x = int(positions[i][0])
        y = int(positions[i][1])
        
        d = dist_maps[i, x, y]
        
        if d == -1:
            penalty += 1e6 
        else:
            total_dist += d
            
    return total_dist + penalty


@dataclass
class GAProParams:
    pop_size: int = 50          
    generations: int = 500 
    mutation: float = 0.1 

    tower_gain: int = 10
    robot_gain: int = 4
    signal_threshold: float = 1.0
    signal_max: float = 10.0
    
    w_obstacle: float = 1e9      
    w_hard_constraint: float = 1e7 
    w_connect: float = 1e5
    w_travel: float = 1
    # too big: overlap at station
    
    init_method: str = "connected" 

    patience: int = 10 
    early_stop: float = 1.0

class GAPro:
    def __init__(self, env: Env, params: GAProParams = None):
        self.env = env
        self.num_robots = env.robots_number 
        self.w, self.h = env.shape
        self.params = params if params else GAProParams()

        if hasattr(self.env, 'map'):
            self.map_data = np.array(self.env.map, dtype=np.int32)
        else:
            self.map_data = np.zeros((self.w, self.h), dtype=np.int32)
            for x in range(self.w):
                for y in range(self.h):
                    if self.env.is_obstacle(x, y):
                        self.map_data[x, y] = 1
        
        self.tower_pos = (float(self.w // 2), float(self.h // 2))
        
        tower_x, tower_y = int(self.tower_pos[0]), int(self.tower_pos[1])
        
        self.dist_maps = np.zeros((self.num_robots, self.w, self.h), dtype=np.int32)
       
        base_dist_map = jit_bfs_distance_map(self.map_data, tower_x, tower_y)
        
        for i in range(self.num_robots):
            self.dist_maps[i] = base_dist_map

        margin = int(self.params.robot_gain)
        self.safe_x_range = (margin, self.w - margin - 1)
        self.safe_y_range = (margin, self.h - margin - 1)
        
        try:
            threshold = max(1e-3, self.params.signal_threshold)
            term = 20.0 / threshold
            physics_factor = 0.0 if term <= 1.0 else math.sqrt(0.5 * math.log(term))
        except ValueError:
            physics_factor = 1.0
        self.connect_factor = min(1.0, physics_factor)
        
        self.tower_range_sq = ((self.params.tower_gain + self.params.robot_gain) * self.connect_factor) ** 2
        self.robot_range_sq = ((self.params.robot_gain + self.params.robot_gain) * self.connect_factor) ** 2

    def get_valid_position(self, center: Tuple[int, int] = None, radius: int = None) -> Tuple[int, int]:
        min_x, max_x = self.safe_x_range
        min_y, max_y = self.safe_y_range
        for _ in range(20):
            if center and radius:
                nx = center[0] + random.randint(-radius, radius)
                ny = center[1] + random.randint(-radius, radius)
                rx = max(min_x, min(max_x, nx))
                ry = max(min_y, min(max_y, ny))
            else:
                rx = random.randint(min_x, max_x)
                ry = random.randint(min_y, max_y)
            if self.map_data[rx, ry] == 0:
                return (rx, ry)
        return (random.randint(min_x, max_x), random.randint(min_y, max_y))
    
    def cobstacle(self, positions):
        cost = 0
        min_x, max_x = self.safe_x_range
        min_y, max_y = self.safe_y_range
        for i in range(len(positions)):
            x, y = positions[i]
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                cost += self.params.w_obstacle
                continue
            if self.map_data[int(x), int(y)] == 1:
                cost += self.params.w_obstacle
        return cost
    
    def cconnectivity(self, positions):
        n = len(positions)
        connected = [False] * n
        queue = []
        tx, ty = self.tower_pos
        
        for i, (px, py) in enumerate(positions):
            if (px - tx)**2 + (py - ty)**2 < self.tower_range_sq:
                connected[i] = True
                queue.append(i)
                
        if not queue:
            dist_pen = sum(math.sqrt((p[0]-tx)**2 + (p[1]-ty)**2) * 100 for p in positions)
            return self.params.w_hard_constraint + dist_pen, connected

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

    def ctravel(self, positions):
        pos_arr = np.array(positions, dtype=np.int32)
        return jit_calculate_travel_cost(pos_arr, self.dist_maps) * self.params.w_travel

    def evaluate(self, positions: List[Tuple[int, int]]) -> float:
        float_positions = [(float(x), float(y)) for x, y in positions]
        
        cost_obs = self.cobstacle(float_positions)
        if cost_obs > 0: return cost_obs + self.params.w_hard_constraint
            
        cost_connect, connected = self.cconnectivity(float_positions)
        if cost_connect > 0: return cost_obs + cost_connect + self.params.w_hard_constraint
        
        cost_travel = self.ctravel(positions)

        pos_arr = np.array(float_positions, dtype=np.float64)
        conn_arr = np.array(connected, dtype=np.bool_)
        cost_cover = jit_calculate_coverage(
            self.map_data, pos_arr, conn_arr, self.tower_pos,
            float(self.params.tower_gain), float(self.params.robot_gain), 
            float(self.params.signal_max), float(self.params.signal_threshold)
        )
        
        return cost_obs + cost_connect + cost_travel + cost_cover

    def process(self) -> List[Tuple[int, int]]:
        population = self.init_population()
        chosen_solution = None
        chosen_cost = float('inf') 
        stall = 0; best_hist = float('inf')
        
        for gen in range(self.params.generations):
            costs = [self.evaluate(ind) for ind in population]
            min_c = min(costs)
            
            if min_c < chosen_cost:
                chosen_cost = min_c
                chosen_solution = population[costs.index(min_c)]
            
            if (best_hist - chosen_cost) > self.params.early_stop:
                stall = 0; best_hist = chosen_cost
            else: stall += 1
                
            if stall >= self.params.patience:
                print(f"Early stop at Gen {gen+1}, Cost: {chosen_cost:.2f}")
                break

            new_pop = [chosen_solution] 
            while len(new_pop) < self.params.pop_size:
                p1 = self.parent_select(population, costs)
                p2 = self.parent_select(population, costs)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1); self.mutate(c2)
                new_pop.extend([c1, c2])
            population = new_pop[:self.params.pop_size]
        
        print(f"End. Best Cost: {chosen_cost:.2f}")
        return chosen_solution if chosen_solution else population[0]

    def init_population(self) -> List[List[Tuple[int, int]]]:
        pop = []
        is_conn = (self.params.init_method == "connected")
        factor = self.connect_factor if is_conn else 1.0
        init_r = int(self.params.tower_gain * 2 * factor) if is_conn else None
        center = (int(self.tower_pos[0]), int(self.tower_pos[1])) if is_conn else None

        for _ in range(self.params.pop_size):
            ind = []
            for _ in range(self.num_robots):
                ind.append(self.get_valid_position(center, init_r))
            pop.append(ind)
        return pop

    def mutate(self, ind):
        if random.random() < self.params.mutation:
            idx = random.randint(0, len(ind) - 1)
            if random.random() < 0.7: 
                ind[idx] = self.get_valid_position(center=ind[idx], radius=3)
            else:
                ind[idx] = self.get_valid_position() 

    def parent_select(self, pop, scores, k=3):
        idxs = random.sample(range(len(pop)), k)
        return pop[min(idxs, key=lambda i: scores[i])]
    
    def crossover(self, p1, p2):
        if len(p1) < 2: return list(p1), list(p2)
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]