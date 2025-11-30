import math
import heapq
from typing import List, Tuple, Set, Dict, Optional
from core import Env
from .node import State, Path

DIRECTIONS = [ (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0) ]
COST_STRAIGHT = 1.0
COST_DIAGONAL = math.sqrt(2)

class Planner():
    def __init__(self, env: Env):
        self.env = env
        self.mid_x = self.env.w // 2
        self.mid_y = self.env.h // 2
        
        self.starts = [(self.mid_x, self.mid_y) for _ in range(self.env.robots_number)]

    
    def process(self, goals: List[Tuple[int, int]]) -> List[Path]:
        raise NotImplementedError

    def check_vertex_conflict(
        self,
        neighbor_state: State, 
        constraints_list: Set[State]
    ) -> bool:
        if neighbor_state.x == self.mid_x and neighbor_state.y == self.mid_y:
            return False
        return neighbor_state in constraints_list

    def check_swap_conflict(
        self,
        current_state: State, 
        neighbor_state: State, 
        constraints_list: Set[State]
    ) -> bool:
        if (current_state.x == self.mid_x and current_state.y == self.mid_y) or \
           (neighbor_state.x == self.mid_x and neighbor_state.y == self.mid_y):
            return False
        swap_state = State(t=neighbor_state.t, x=current_state.x, y=current_state.y)
        prev_swap_state = State(t=current_state.t, x=neighbor_state.x, y=neighbor_state.y)
        return swap_state in constraints_list and prev_swap_state in constraints_list

    @staticmethod
    def check_corner_cutting(current: Tuple[int, int], next: Tuple[int, int], env: Env) -> bool:
        cx, cy = current
        nx, ny = next
        
        dx = nx - cx
        dy = ny - cy
            
        # neighbor 1
        if env.is_obstacle(cx + dx, cy):
            return True
            
        # neighbor 2
        if env.is_obstacle(cx, cy + dy):
            return True
            
        return False


    @staticmethod
    def heuristic(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        
        return (COST_STRAIGHT * (max(dx, dy) - min(dx, dy))) + (COST_DIAGONAL * min(dx, dy))

    @staticmethod
    def is_valid_location(x: int, y: int, env: Env) -> bool:
        return 0 <= x < env.w and 0 <= y < env.h and not env.is_obstacle(x, y)

    @staticmethod
    def calculate_total_path_cost(solution: List[Path]) -> int:
        total_cost = 0
        for path in solution:
            if path:
                total_cost += len(path) - 1 
        return total_cost
    
    def state_time_a_star(
        self,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        constraints_list: Set[State],  
        env: Env,
        check_collisions: bool,  
        max_time: int = 100
    ) -> Optional[Path]:
        
        start_state = State(t=0, x=start_pos[0], y=start_pos[1])
        
        open_set = []
        heapq.heappush(open_set, (Planner.heuristic(start_pos, goal_pos), start_state)) 
        
        came_from: Dict[State, State] = {}
        g_score: Dict[State, float] = {start_state: 0}

        while open_set:
            _, current_state = heapq.heappop(open_set)
            
            if (current_state.x, current_state.y) == goal_pos:
                path = []
                while current_state in came_from:
                    path.append(current_state)
                    current_state = came_from[current_state]
                path.append(start_state)
                return path[::-1] 

            if current_state.t > max_time:
                continue
                
            for dx, dy in DIRECTIONS:
                neighbor_x, neighbor_y = current_state.x + dx, current_state.y + dy
                neighbor_t = current_state.t + 1
                neighbor_state = State(t=neighbor_t, x=neighbor_x, y=neighbor_y)
                
                if not Planner.is_valid_location(neighbor_x, neighbor_y, env):
                    continue
                
                if dx != 0 and dy != 0:
                    if Planner.check_corner_cutting((current_state.x, current_state.y), (neighbor_x, neighbor_y), env):
                        continue
         
                if check_collisions:
                    if self.check_vertex_conflict(neighbor_state, constraints_list):
                        continue
                    
                    if self.check_swap_conflict(current_state, neighbor_state, constraints_list):
                        continue

                if dx != 0 and dy != 0:
                    move_cost = COST_DIAGONAL
                else:
                    move_cost = COST_STRAIGHT 
                
                new_g_score = g_score[current_state] + move_cost

                if neighbor_state not in g_score or new_g_score < g_score[neighbor_state]:
                    g_score[neighbor_state] = new_g_score
                    f_score = new_g_score + Planner.heuristic((neighbor_x, neighbor_y), goal_pos)
                    heapq.heappush(open_set, (f_score, neighbor_state))
                    came_from[neighbor_state] = current_state
                    
        return None 