import os
import sys
import time  

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# from multi_robot import World, Visualizer as OldVisualizer, Robot, Entity, Obstacle
from wifi_signal import SignalVisualizer
from core.env import Env
from placement.pso import ParticleSwarmOptimizer, PSOParams
from placement.ga import GA, GAParams
from planner.prioritize import PrioritizedPlanner
from planner.cbs import CBSPlanner
from planner.jss import JointAStarPlanner
from visualizer import Visualizer as BaseNewVisualizer
from placement.node import get_valid_position_on_map

def run_coverage(num):
    env = Env(map_path)
    if num > 0:
        env.set_robots_number(num)
    targets = []
    cost = 0.0
    actual_iter = 0 
    max_iter_setting = 0 
    config_summary = {
        "Map": selected_map,
        "Algorithm": cov_algo,
        "Stage": "Coverage",
        "Init Method": cov_init_method
    }
    
    if cov_algo == "GA":
        config_summary.update({
            "Pop Size": ga_pop_size,
            "Generations": ga_generations
        })
        ga_params = GAParams(
            generations=ga_generations, 
            pop_size=ga_pop_size, 
            init_method=cov_init_method
        )
        ga = GA(env, params=ga_params)
        targets = ga.process()
        # Re-evaluate to get the cost for display
        if targets:
            cost = ga.evaluate(targets)
            actual_iter = ga.stopped_iter
            max_iter_setting = ga_generations
        return env, targets, cost, config_summary, actual_iter, max_iter_setting 
        
    elif cov_algo == "PSO":
        config_summary.update({
            "Particles": pso_particle_count,
            "Max Iter": pso_max_iter,
            "Repulsion": pso_repulsion
        })
        pso_params = PSOParams(
            particle_count=pso_particle_count,
            max_iter=pso_max_iter,
            use_collision=pso_repulsion,
            init_method=cov_init_method
        )
        pso = ParticleSwarmOptimizer(env, params=pso_params) 
        targets = pso.process()
        # Re-evaluate to get the cost for display
        if targets:
            cost = pso.evaluate(targets)
            actual_iter = pso.stopped_iter
            max_iter_setting = pso_max_iter
        
    return env, targets, cost, config_summary, actual_iter, max_iter_setting

def get_available_maps():
    maps_dir = os.path.join(parent_dir, "maps")
    if not os.path.exists(maps_dir):
        return []
    files = [f for f in os.listdir(maps_dir) if f.endswith(".txt")]
    return sorted(files)

class StreamlitNewVisualizer(BaseNewVisualizer):
    def _set_consistent_style(self, ax, fig_width):
        base_width = 5.0
        base_fontsize = 9.0
        scale = fig_width / base_width
        scaled_fontsize = base_fontsize * scale
        ax.tick_params(axis='both', which='major', labelsize=scaled_fontsize)
        return scaled_fontsize

    def get_points_fig(self, points):
        fig, ax = self._prepare_ax()
        current_size = (3, 3)
        fig.set_size_inches(current_size)
        font_size = self._set_consistent_style(ax, current_size[0])
        cmap = matplotlib.colormaps["tab20"]

        ax.scatter(self.w / 2, self.h / 2, marker='x', color='gray', s=80, linewidths=2)
        circle = plt.Circle((self.w / 2, self.h / 2), 10, edgecolor='gray', fill=False, linewidth=1.5)
        ax.add_patch(circle)

        for i, p in enumerate(points):
            x, y = p
            ax.scatter(x, y, color=cmap(i), s=60, label=f'robot {i + 1}')
            circle = plt.Circle((x, y), radius=4, edgecolor=cmap(i), fill=False, linewidth=1.5)
            ax.add_patch(circle)
        
        # add connected line
        nodes, edges = self._compute_connectivity(points)
        for i, j in edges:
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            ax.plot([x1, x2], [y1, y2], color='black', linewidth=1, alpha=0.5)
        
        return fig

    def get_animation_html(self, paths, interval=200):
        fig, ax = self._prepare_ax()
        current_size = (5, 5)
        fig.set_size_inches(current_size)
        font_size = self._set_consistent_style(ax, current_size[0])
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

class StreamlitSignalVisualizer(SignalVisualizer):
    def get_animation_html(self, paths, interval=200):
        fig, ax = self._prepare_ax()
        fig.set_size_inches(8, 8)

        robot_colors = matplotlib.colormaps["tab20"]
        self.robot_plots = []
        
        for i in range(len(paths)):
            plot, = ax.plot([], [], marker='o', markersize=8, color=robot_colors(i), label=f"Robot {i}")
            self.robot_plots.append(plot)
            
        fig.colorbar(self.heatmap, ax=ax, label="Signal Strength")
        ax.legend(loc="upper right")
        
        num_frames = max(len(p) for p in paths) if paths else 0
        
        def update(frame):
            robot_positions = []
            for path in paths:
                idx = min(frame, len(path) - 1)
                robot_positions.append((path[idx].x, path[idx].y))
            
            for i, pos in enumerate(robot_positions):
                if i < len(self.robot_plots):
                    self.robot_plots[i].set_data([pos[0]], [pos[1]])
            
            Z = self._compute_signal(robot_positions)
            self.heatmap.set_array(Z.ravel())
            
            return self.robot_plots + [self.heatmap, self.station_plot]
        
        ani = FuncAnimation(
            fig, update,
            frames=num_frames,
            interval=interval,
            blit=False,
            repeat=True
        )
        
        return ani.to_jshtml(default_mode='loop')
    
matplotlib.use('Agg')
plt.rcParams['animation.embed_limit'] = 100.0 
st.set_page_config(layout='wide', page_title="Robot Planning Demo")
st.title('Multi-Robot Planning for Communication Coverage Optimization')


st.sidebar.title('Settings')
st.sidebar.header("Visualization options")
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

# Default values
pso_particle_count = 100
pso_max_iter = 200
pso_repulsion = False
cov_init_method = "connected"

# GA Defaults
ga_generations = 200
ga_pop_size = 50

# JSS Defaults
Jss_num = 2

# other planned 
target_num = 0

if task_type in ["Coverage", "Both (Stage 1 + Stage 2)"]:
    st.sidebar.subheader("Stage 1: Coverage")
    target_mode = st.sidebar.radio(
            "Number to generate",
            ["Default with map", "Custom Number"]
        )
        
    if target_mode == "Custom Number":
        target_num = st.sidebar.slider("Number of Targets", min_value=1, max_value=30, value=10)
    else:
        target_num = 0
        
    cov_init_method = st.sidebar.selectbox(
        "Initialization Mode", 
        ["connected", "random"], 
        key="cov_init"
    )
    
    cov_algo = st.sidebar.selectbox("Coverage Algorithm", ["PSO", "GA"], key="cov")
    
    with st.sidebar.expander(f"{cov_algo} Parameters Options", expanded=True):
        if cov_algo == "PSO":
            pso_particle_count = st.slider("Particle Count", min_value=50, max_value=100, value=pso_particle_count)
            pso_max_iter = st.slider("Max Iterations", min_value=50, max_value=500, value=pso_max_iter)
            pso_repulsion = st.checkbox("Enable Repulsion", value=pso_repulsion)
            
        elif cov_algo == "GA":
            ga_generations = st.slider("Max Iterations", min_value=50, max_value=500, value=ga_generations)
            ga_pop_size = st.slider("Population Size", min_value=50, max_value=100, value=ga_pop_size)
if task_type in ["Planner", "Both (Stage 1 + Stage 2)"]:
    st.sidebar.subheader("Stage 2: Planning")
    plan_algo = st.sidebar.selectbox("Choose Algorithm", ["PP", "CBS", "JSS"], key="plan")
    
    with st.sidebar.expander(f"{plan_algo} Parameters", expanded=True):
        if plan_algo == "PP":
            priority_mode = st.sidebar.radio("Priority Mode",["default", "random", "closest", "far"])
            if task_type not in ["Both (Stage 1 + Stage 2)"]:
                target_mode = st.sidebar.radio(
                        "Number to generate",
                        ["Default with map", "Custom Number"]
                    )
                    
                if target_mode == "Custom Number":
                    target_num = st.sidebar.slider("Number of Targets", min_value=1, max_value=25, value=5)
                else:
                    target_num = 0

        if plan_algo == "CBS":    
            target_mode = st.sidebar.radio(
                    "Number to generate",
                    ["Default with map", "Custom Number"]
                )
            if task_type not in ["Both (Stage 1 + Stage 2)"]:
                if target_mode == "Custom Number":
                    target_num = st.sidebar.slider("Number of Targets", min_value=1, max_value=25, value=5)
                else:
                    target_num = 0
        if plan_algo == "JSS":
           if task_type not in ["Both (Stage 1 + Stage 2)"]:
               Jss_num = st.slider("Number to generate", min_value=1, max_value=3, value=Jss_num)

btn_col, txt_col = st.columns([1,15], vertical_alignment="center")
with btn_col:
    run_pressed = st.button("Run", type="primary")

if run_pressed:
    status_msg = f"Running **{task_type}**"
    if cov_algo: status_msg += f" with **{cov_algo}**"
    if plan_algo: status_msg += f" with **{plan_algo}**"
    
    with txt_col:
        st.info(status_msg)
    if task_type == "Coverage":
        with st.spinner("Simulating Coverage..."):
            start_time = time.time() 
            env, targets, best_cost, config, stop_it, max_iter = run_coverage(target_num)
            target_num = 0
            exec_time = time.time() - start_time
        if targets:
            col1, col2 = st.columns([2, 2]) 
            
            with col1:
                viz = StreamlitNewVisualizer(env)
                fig = viz.get_points_fig(targets)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
            
            with col2:                
                st.markdown("#### ⚙️ Configuration")
                st.json(config, expanded=True)

                st.markdown("#### 📊 Performance")
                desired = env.robot_num
                found = len(targets)
                st.write(f"**Iterations:** {stop_it} " )
                st.write(f"**Robot Number:** {found} / {desired}")
                st.write(f"**Runtime:** {exec_time:.4f} s")
                st.write(f"**Overall Cost:** {best_cost:.4f}")
                
                st.markdown("#### 📍 Targets Found")
                target_str = "\n".join([str(t) for t in targets])
                st.text(target_str)

        else:
            st.error("No targets found.")

    elif task_type == "Planner":        
            if not os.path.exists(map_path):
                st.error(f"Can not find map file: {map_path}")
            else:
                env = Env(map_path)
                if plan_algo == "JSS":
                    env.set_robots_number(Jss_num)
                else:
                    if target_num > 0:
                        env.set_robots_number(target_num)
                        target_num = 0
                num_robots = env.robot_num
                # Random generate targets
                demo_targets = []
                
                # Define map ranges (using entire map)
                x_range = (0, env.w - 1)
                y_range = (0, env.h - 1)
                
                # Identify start positions to avoid overlaps if necessary
                starts = []
                if hasattr(env, 'agents'):
                    starts = [tuple(a.pos) for a in env.agents if hasattr(a, 'pos')]
                elif hasattr(env, 'starts'):
                    starts = env.starts

                # Generation Loop
                while len(demo_targets) < num_robots:
                    # Get a valid random position (legal: inside map, not obstacle)
                    raw_pos = get_valid_position_on_map(np.array(env.map, dtype=np.float32), x_range, y_range)
                    target = (int(raw_pos[0]), int(raw_pos[1]))
                    
                    if target not in demo_targets and target not in starts:
                        demo_targets.append(target)
                    
                
                if num_robots < len(demo_targets):
                    demo_targets = demo_targets[:num_robots]
                
                paths = []
                start_time = time.time()

                with st.spinner("Planning Paths..."):
                    if plan_algo == "PP":
                        planner = PrioritizedPlanner(env, priority_mode=priority_mode)
                        paths = planner.process(demo_targets)
                    elif plan_algo == "CBS":
                        planner = CBSPlanner(env, max_time=100)
                        paths = planner.process(demo_targets)
                    elif plan_algo == "JSS":
                        planner = JointAStarPlanner(env)
                        paths = planner.process(demo_targets)
                
                exec_time = time.time() - start_time

                if paths:
                    col1, col2 = st.columns([2, 2])
                    
                    with col1:
                        viz = StreamlitNewVisualizer(env)
                        html_code = viz.get_animation_html(paths, interval=200)
                        components.html(html_code, height=900, scrolling=True)

                    with col2:
                        st.markdown("#### 📍 Targets Found")
                        demo_target_str = "\n".join([str(t) for t in demo_targets])
                        st.text(demo_target_str)
                        
                        st.markdown("#### ⚙️ Configuration")
                        st.write(f"Algo: {plan_algo}")
                        st.write(f"Mode: {priority_mode}")
                        st.write(f"Map: {selected_map}")
                        
                        st.markdown("#### 📊 Performance")
                        total_steps = sum([len(p) for p in paths])
                        makespan = max([len(p) for p in paths])
                        avg_len = total_steps / len(paths)

                        st.write(f"Total Steps: {total_steps}")
                        st.write(f"Makespan: {makespan}")
                        st.write(f"Avg Cost: {avg_len:.1f}")
                        st.write(f"Time: {exec_time:.4f} s")                  
                       


    elif task_type == "Both (Stage 1 + Stage 2)":
        with st.spinner("Simulating Integrated System..."):
            start_time = time.time()
            env, targets, best_cost, config, stop_it, max_iter = run_coverage(target_num)
            target_num = 0
            cov_time = time.time() - start_time

            paths = []
            if targets:
                with st.spinner("Planning Paths..."):
                    if plan_algo == "PP":
                        planner = PrioritizedPlanner(env, priority_mode=priority_mode)
                        paths = planner.process(targets)
                    elif plan_algo == "CBS":
                        planner = CBSPlanner(env, max_time=100)
                        paths = planner.process(targets)
                    elif plan_algo == "JSS":
                        if env.robot_num < 4:
                            planner = JointAStarPlanner(env)
                            paths = planner.process(targets)
                        else: 
                            st.error("JSS only supports up to 3 robots currently.")
            
            total_time = time.time() - start_time

            col1, col2 = st.columns([3, 1])
            
            with col1:
                if paths:
                    sig_viz = StreamlitSignalVisualizer(env, resolution=80) 
                    html_code = sig_viz.get_animation_html(paths, interval=200)
                    
                    components.html(html_code, height=800, scrolling=False)
                    plt.close(sig_viz.heatmap.figure) 
                else:
                    st.warning("No paths generated, cannot run signal simulation.")
                

            with col2:
                st.subheader("Coverage Evaluation")

                st.markdown("#### ⚙️ Configuration")
                st.json(config, expanded=True)
                st.markdown("#### 📊 Coverage Results")
                st.write(f"**Robots:** {len(targets)} / {env.robot_num}")
                st.write(f"**Cost:** {best_cost:.2f}")
                st.write(f"**Stage 1 Time:** {cov_time:.4f} s")

                st.markdown("#### 🚀 Planner Results")
                total_steps = sum([len(p) for p in paths])
                st.write(f"**Total Steps:** {total_steps}")
                st.write(f"**Total Time:** {total_time:.4f} s")

                st.markdown("#### 📍 Targets Found")
                target_str = "\n".join([str(t) for t in targets])
                st.text(target_str)

            col3, col4 = st.columns([2, 2])

            with col3:
                viz = StreamlitNewVisualizer(env)
                fig = viz.get_points_fig(targets)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

            with col4:
                viz = StreamlitNewVisualizer(env)
                if plan_algo == "JSS":
                    # error fot num
                    if env.robot_num < 4:
                        html_code = viz.get_animation_html(paths, interval=200)
                        components.html(html_code, height=900, scrolling=True)
                else:
                    html_code = viz.get_animation_html(paths, interval=200)
                    components.html(html_code, height=900, scrolling=True)

               