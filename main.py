import argparse

from core import Env
from planner import PrioritizedPlanner
from placement import ParticleSwarmOptimizer, GA
from visualizer import Visualizer, SignalVisualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='filepath to map file')
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    env = Env(args.file)
    cov_opt = GA(env)
    points = cov_opt.process()

    cost_obs, cost_connect, cost_coll, cost_travel, cost_cover, cost_all = cov_opt.compute_metric()
    

    planner = PrioritizedPlanner(env, 'far')
    paths = planner.process(points)

    viz = SignalVisualizer(env)
    viz.animate(paths, args.output)
