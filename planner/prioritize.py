import random
from typing import List, Tuple, Set
from core import Env
from .base import Planner
from .node import State, Path

class PrioritizedPlanner(Planner):    
    def __init__(self, env: Env, priority_mode: str = "default"):
        super().__init__(env)
        self.priority_mode = priority_mode
        print(f"PP-Priority Mode: {self.priority_mode}")

    def _get_planning_order(self, goals: List[Tuple[int, int]]) -> List[int]:
        num_robots = len(self.starts)
        indices = list(range(num_robots)) 
        station_pos = (0, 0) 

        if self.priority_mode == "random":
            random.shuffle(indices)
            return indices
            
        elif self.priority_mode == "far":
            indices.sort(
                key=lambda i: self.heuristic(station_pos, goals[i]), 
                reverse=True
            )
            return indices

        elif self.priority_mode == "closest":
            indices.sort(
                key=lambda i: self.heuristic(station_pos, goals[i]),
                reverse=False
            )
            return indices
            
        else: 
            print(" default")
            return indices

    def process(self, goals: List[Tuple[int, int]]) -> List[Path]:
        
        if len(self.starts) != len(goals):
            raise ValueError(f"number incompatible")
        
        all_paths: List[Path] = [None] * len(self.starts)
        reservation: Set[State] = set()
        planning_order = self._get_planning_order(goals)
        
        for i in planning_order:
            start_pos = self.starts[i]
            goal_pos = goals[i]
            
            path = self.state_time_a_star(
                start_pos,
                goal_pos,
                reservation,
                self.env,
                check_collisions=True 
            )
            
            if path is None:
                print("failed to find path for all robots")
                return [] 
            
            all_paths[i] = path
            
            for state in path:
                reservation.add(state)
            
            last_state = path[-1]
            max_time = 400 # weird
            for t in range(last_state.t + 1, max_time + 1):
                reservation.add(State(t, last_state.x, last_state.y))

        print("Prioritized Planning SUCCESS.")
        return all_paths