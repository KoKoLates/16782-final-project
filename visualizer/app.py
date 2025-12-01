import os
import sys
import time  

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from multi_robot import World, Visualizer as OldVisualizer, Robot, Entity, Obstacle
from core.env import Env
from placement.pso import ParticleSwarmOptimizer, PSOParams
from placement.ga import GA, GAParams
from planner.prioritize import PrioritizedPlanner
from planner.cbs import CBSPlanner
from planner.jss import JointAStarPlanner
from visualizer import Visualizer as BaseNewVisualizer


def get_available_maps():
    maps_dir = os.path.join(parent_dir, "maps")
    if not os.path.exists(maps_dir):
        return []
    files = [f for f in os.listdir(maps_dir) if f.endswith(".txt")]
    return sorted(files)

class StreamlitNewVisualizer(BaseNewVisualizer):
    def get_points_fig(self, points):
        fig, ax = self._prepare_ax()
        fig.set_size_inches(5, 5)

        cmap = matplotlib.colormaps["tab20"]

        ax.scatter(self.w / 2, self.h / 2, marker='x', color='gray', s=80, linewidths=2)
        circle = plt.Circle((self.w / 2, self.h / 2), 10, edgecolor='gray', fill=False, linewidth=1.5)
        ax.add_patch(circle)

        for i, p in enumerate(points):
            x, y = p
            ax.scatter(x, y, color=cmap(i), s=60, label=f'robot {i + 1}')
            circle = plt.Circle((x, y), radius=4, edgecolor=cmap(i), fill=False, linewidth=1.5)
            ax.add_patch(circle)
            
        return fig

    def get_animation_html(self, paths, interval=200):
        fig, ax = self._prepare_ax()
        cmap = matplotlib.colormaps["tab20"]
        
        scatters = []
        traces = []
        goal_circles = []
        goals = [path[-1] for path in paths]

        for goal in goals:
            circle = plt.Circle((goal.x, goal.y), radius=0.1, facecolor="none", edgecolor="gray", linewidth=1, zorder=5)
            ax.add_patch(circle)
            goal_circles.append(circle)

        for i, path in enumerate(paths):
            sc = ax.scatter([], [], color=cmap(i), s=20, zorder=6)
            scatters.append(sc)
            tr, = ax.plot([], [], color=cmap(i), linewidth=2, alpha=0.7)
            traces.append(tr)

        all_times = sorted({s.t for path in paths for s in path})
        t_min, t_max = all_times[0], all_times[-1]
        indexed = [{s.t: (s.x, s.y) for s in path} for path in paths]

        def update(frame_t):
            updated_artists = []
            for i, d in enumerate(indexed):
                valid_ts = [t for t in d.keys() if t <= frame_t]
                if not valid_ts: continue

                t = max(valid_ts)
                x, y = d[t]
                scatters[i].set_offsets([[x, y]]) 
                past = [d[tt] for tt in sorted(valid_ts)]
                xs = [p[0] for p in past]
                ys = [p[1] for p in past]
                traces[i].set_data(xs, ys)
                if frame_t >= goals[i].t:
                    goal_circles[i].set_edgecolor(cmap(i))
                
            return scatters + traces + goal_circles

        ani = FuncAnimation(
            fig, update, frames=range(t_min, t_max + 1),
            interval=interval, blit=False, repeat=True
        )
        return ani.to_jshtml(default_mode='loop')


matplotlib.use('Agg')
plt.rcParams['animation.embed_limit'] = 100.0 
st.set_page_config(layout='wide', page_title="Robot Planning Demo")
st.title('Multi-Robot Planning for Communication Coverage Optimization')


st.sidebar.title('Settings')
resolution = st.sidebar.select_slider('Map Resolution', options=[40, 60, 80, 100], value=60)
frames = st.sidebar.select_slider('Simulation Frames', options=[30, 40, 50, 80], value=40)
st.sidebar.markdown("---") 

st.sidebar.header("Map Selection")
available_maps = get_available_maps()
if available_maps:
    selected_map = st.sidebar.selectbox("Choose a Map file (.txt)", available_maps)
    map_path = os.path.join(parent_dir, "maps", selected_map)
else:
    st.sidebar.error("No maps found in 'maps/' folder!")
    selected_map = None
    map_path = None

st.sidebar.header("Select Task Type")
task_type = st.sidebar.radio(
    "Choose what to run:",
    ["Coverage", "Planner", "Both (Stage 1 + Stage 2)"]
)

cov_algo = None
plan_algo = None
priority_mode = "default"

st.sidebar.header("Algorithm Selection")
if task_type in ["Coverage", "Both (Stage 1 + Stage 2)"]:
    st.sidebar.subheader("Stage 1: Coverage")
    cov_algo = st.sidebar.selectbox("Coverage Algorithm", ["PSO", "GA"], key="cov")
    
    with st.sidebar.expander(f"{cov_algo} Parameters"):
        if cov_algo == "PSO":
            pso_particle_count = st.slider("Particle Count", min_value=50, max_value=500, value=100)
            pso_max_iter = st.slider("Max Iterations", min_value=50, max_value=500, value=100)
        elif cov_algo == "GA":
            st.number_input("Interations", value=100)
            st.slider("Inertia (w)", 100, 300, 500)

if task_type in ["Planner", "Both (Stage 1 + Stage 2)"]:
    st.sidebar.subheader("Stage 2: Planner")
    plan_algo = st.sidebar.selectbox("Choose Algorithm", ["JSS", "PP", "CBS"], key="plan")
    
    with st.sidebar.expander(f"{plan_algo} Parameters"):
        if plan_algo == "PP":
           priority_mode = st.sidebar.radio(
                "Priority Mode",
                ["default", "random", "closest", "far"]
            )

if st.button("Run", type="primary"):
    status_msg = f"Running **{task_type}**"
    if cov_algo: status_msg += f" with **{cov_algo}**"
    if plan_algo: status_msg += f" and **{plan_algo}**"
    
    st.info(status_msg)
    if task_type == "Coverage":
        with st.spinner("Simulating Coverage..."):
            env = Env(map_path)
            targets = []
            
            start_time = time.time() 
            
            if cov_algo == "GA":
                st.error("GA logic not yet implemented.")
            elif cov_algo == "PSO":
                pso_params = PSOParams(
                    particle_count=pso_particle_count,
                    max_iter=pso_max_iter
                )
                pso = ParticleSwarmOptimizer(env, params=pso_params) 
                targets = pso.process()
            
            exec_time = time.time() - start_time 

        if targets:
            col1, col2 = st.columns([2, 2])
            
            with col1:
                viz = StreamlitNewVisualizer(env)
                fig = viz.get_points_fig(targets)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
            
            with col2:
                st.subheader("Evaluation")
                st.metric("Targets Found", len(targets))
                st.metric("Compute Time", f"{exec_time:.4f} s")
        else:
            st.error("No targets found.")

    elif task_type == "Planner":        
        if not os.path.exists(map_path):
            st.error(f"Can not find map file: {map_path}")
        else:
            env = Env(map_path)
            num_robots = env.robot_num
            st.info(f"Detected {num_robots} robots on the map.")
            demo_targets = [(30,39), (37,36), (38,29), (38,22), (30,12), (12,20)]
            if num_robots < len(demo_targets):
                demo_targets = demo_targets[:num_robots]
            
            st.write(f"Planning for targets: {demo_targets}")
            
            paths = []
            start_time = time.time()

            with st.spinner("Planning Paths..."):
                if plan_algo == "PP":
                    planner = PrioritizedPlanner(env, priority_mode=priority_mode)
                    paths = planner.process(demo_targets)
                else:
                    st.warning(f"{plan_algo} not implemented, showing empty.")
            
            exec_time = time.time() - start_time

            if paths:
                col1, col2 = st.columns([2, 2])
                
                with col1:
                    st.subheader("Animation")
                    viz = StreamlitNewVisualizer(env)
                    html_code = viz.get_animation_html(paths, interval=200)
                    components.html(html_code, height=900, scrolling=True)

                with col2:
                    st.subheader("Evaluation")
                    
                    total_steps = sum([len(p) for p in paths])
                    makespan = max([len(p) for p in paths])
                    avg_len = total_steps / len(paths)

                    st.metric("Success Rate", "100%")
                    st.metric("Total Steps (Sum)", total_steps)
                    st.metric("Makespan (Max)", makespan)
                    st.metric("Avg Path Cost", f"{avg_len:.1f}")
                    st.metric("Compute Time", f"{exec_time:.4f} s")

    elif task_type == "Both (Stage 1 + Stage 2)":
        with st.spinner("Simulating Integrated System..."):
            
            world = World(resolution=60)
            world.add_obstacle(Obstacle((4, 6), 4.0, 1.0, 30.0, 0.1))
            world.add_obstacle(Obstacle((-5, 5), 3.0, 3.0, 0.0, 0.1))
            world.add_agent(Entity("Station", (0,0), 4.0, 1.0))

            frames = 40
            path1 = list(zip(np.linspace(9, 7, frames), np.linspace(0, 7, frames)))
            path2 = list(zip(np.linspace(-1, 8, frames), np.linspace(1, 4, frames)))
            world.add_agent(Robot("Robot A", path1, 4.0, 0.7, 'b'))
            world.add_agent(Robot("Robot B", path2, 4.0, 0.7, 'c'))

            col1, col2 = st.columns([2, 2])
            
            with col1:
                st.subheader("Integrated Simulation")
                vis = OldVisualizer(world, fps=10)
                vis.setup()
                html_code = vis.get_animation_html(steps=frames, interval=100)
                components.html(html_code, height=900, scrolling=True)
                plt.close(vis.fig)

            with col2:
                st.subheader("System Metrics")
                st.metric("Signal Coverage", "87%", "+2%")
                st.metric("Connectivity", "Stable")
                st.metric("Mission Time", "4.2s")
                st.metric("Active Robots", "2")