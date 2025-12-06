from typing import List, NamedTuple

class State(NamedTuple):
    t: int
    x: int
    y: int

class Constraint(NamedTuple):
    robot_id: int
    state: State

Path = List[State]
