import numpy as np
import random
import math
from typing import List, Tuple
from core import Env
from dataclasses import dataclass
from .node import get_valid_position_on_map
from .base import (
    Coverage,
    jit_calculate_coverage, 
    jit_calculate_travel_cost, 
    calculate_physics_factor
)


@dataclass
class GAParams:
    pop_size: int = 50          
    generations: int = 200 
    mutation: float = 0.1 

    tower_gain: int = 10
    robot_gain: int = 4
    signal_threshold: float = 1.0
    signal_max: float = 10.0
    
    w_obstacle: float = 1e9      
    w_hard_constraint: float = 1e7 
    w_connect: float = 1e5
    w_travel: float = 0.5   
    
    init_method: str = "connected" 
    patience: int = 20 
    early_stop: float = 1.0

class GA(Coverage):
    def __init__(self, env: Env, params: GAParams = None):
        super().__init__(env)
        self.params = params if params else GAParams()
        
        # Setup Ranges
        margin = int(self.params.robot_gain)
        self.safe_x_range = (margin, self.w - margin - 1)
        self.safe_y_range = (margin, self.h - margin - 1)
        
        # Physics
        p_factor = calculate_physics_factor(self.params.signal_threshold)
        self.connect_factor = min(1.0, p_factor)
        
        self.tower_range_sq = ((self.params.tower_gain + self.params.robot_gain) * self.connect_factor) ** 2
        self.robot_range_sq = ((self.params.robot_gain + self.params.robot_gain) * self.connect_factor) ** 2

    def cobstacle(self, positions):
        cost = 0
        min_x, max_x = self.safe_x_range
        min_y, max_y = self.safe_y_range
        for i in range(len(positions)):
            x, y = positions[i]
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                cost += self.params.w_obstacle
                continue
            if self.map_data[int(x), int(y)] >= 1e9:
                cost += self.params.w_obstacle
        return cost
    
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

    def ctravel(self, positions):
        pos_arr = np.array(positions, dtype=np.float64)
        return jit_calculate_travel_cost(pos_arr, self.map_data) * self.params.w_travel

    def evaluate(self, positions: List[Tuple[int, int]]) -> float:
        float_positions = [(float(x), float(y)) for x, y in positions]
        
        cost_obs = self.cobstacle(float_positions)
        if cost_obs > 0: return cost_obs + self.params.w_hard_constraint
            
        cost_connect, connected = self.cconnectivity(float_positions)
        if cost_connect > 0: return cost_obs + cost_connect + self.params.w_hard_constraint
        
        cost_travel = self.ctravel(float_positions)

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
            
            if abs(best_hist - chosen_cost) < self.params.early_stop:
                stall += 1
            else:
                stall = 0
                best_hist = chosen_cost
                
            if stall >= self.params.patience:
                print(f"Early stop at Gen {gen+1}")
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
        
        raw_result = chosen_solution if chosen_solution else population[0]
        final_result_int = []

        for fx, fy in raw_result:
            ix, iy = int(round(fx)), int(round(fy))
            
            is_valid = False
            
            if 0 <= ix < self.w and 0 <= iy < self.h:
                if self.map_data[ix, iy] < 1e9:
                    final_result_int.append((ix, iy))
                    is_valid = True
            
            if not is_valid:
                found_neighbor = False
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < self.w and 0 <= ny < self.h:
                            if self.map_data[nx, ny] < 1e9:
                                final_result_int.append((nx, ny))
                                found_neighbor = True
                                break
                    if found_neighbor: break
                
                if not found_neighbor:
                    final_result_int.append((ix, iy))

        return final_result_int

    def init_population(self) -> List[List[Tuple[int, int]]]:
        pop = []
        is_conn = (self.params.init_method == "connected")
        factor = self.connect_factor if is_conn else 1.0
        init_r = (self.params.tower_gain * 2 * factor) if is_conn else None
        center = self.tower_pos if is_conn else None

        for _ in range(self.params.pop_size):
            ind = []
            for _ in range(self.num_robots):
                pos = get_valid_position_on_map(self.map_data, self.safe_x_range, self.safe_y_range, center, init_r)
                ind.append(pos)
            pop.append(ind)
        return pop

    def mutate(self, ind):
        if random.random() < self.params.mutation:
            idx = random.randint(0, len(ind) - 1)
            if random.random() < 0.7: 
                ind[idx] = get_valid_position_on_map(self.map_data, self.safe_x_range, self.safe_y_range, center=ind[idx], radius=3)
            else:
                ind[idx] = get_valid_position_on_map(self.map_data, self.safe_x_range, self.safe_y_range)

    def parent_select(self, pop, scores, k=3):
        idxs = random.sample(range(len(pop)), k)
        return pop[min(idxs, key=lambda i: scores[i])]
    
    def crossover(self, p1, p2):
        if len(p1) < 2: return list(p1), list(p2)
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]