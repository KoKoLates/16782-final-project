
from core import Env
from planner import Planner
from coverage import CoverageOptimizer
from visualizer import Visualizer


if __name__ == "__main__":
    env = Env("example.txt")

    cov_opt = CoverageOptimizer(env.grid(), env.obstacles())
    targets = cov_opt.process()

    planner = Planner()
    paths = planner.process(targets)

    viz = Visualizer(env)
    viz.plot(paths)