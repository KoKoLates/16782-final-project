import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multi_robot import Entity, Robot, Obstacle, World, Visualizer

matplotlib.use('Agg')
plt.rcParams['animation.embed_limit'] = 100.0 

st.set_page_config(layout='wide')
st.title('Multi-Robot Planning for Communication Coverage Optimization')

# Sidebar
st.sidebar.title('Settings')
resolution = st.sidebar.select_slider('Map Resolution', options=[40, 60, 80, 100], value=60)
frames = st.sidebar.select_slider('Simulation Frames', options=[30, 40, 50, 80], value=40)
st.sidebar.markdown("---") 

st.sidebar.header("1. Select Task Type")
task_type = st.sidebar.radio(
    "Choose what to run:",
    ["Coverage", "Planner", "Both (Stage 1 + Stage 2)"]
)

cov_algo = None
plan_algo = None

st.sidebar.header("2. Algorithm Selection")


if task_type in ["Coverage", "Both (Stage 1 + Stage 2)"]:
    st.sidebar.subheader("Stage 1: Coverage")
    cov_algo = st.sidebar.selectbox("Choose Algorithm", ["PSO", "GA", "Voroni"], key="cov_box")
    
    with st.sidebar.expander(f"{cov_algo} Parameters"):
        if cov_algo == "PSO":
            st.number_input("Interations", value=100)
            st.slider("Inertia (w)", 100, 300, 500)
        elif cov_algo == "GA":
            st.number_input("Interations", value=100)
            st.slider("Inertia (w)", 100, 300, 500)

if task_type in ["Planner", "Both (Stage 1 + Stage 2)"]:
    st.sidebar.subheader("Stage 2: Planner")
    plan_algo = st.sidebar.selectbox("Choose Algorithm", ["JSS", "PP", "CBS"], key="plan_box")
    
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
      
    with st.spinner(f"Calculating... (Res: {resolution}, Frames: {frames})"):
        
        world = World(resolution=resolution)
        
        world.add_obstacle(Obstacle((4, 6), 4.0, 1.0, 30.0, 0.1))
        world.add_obstacle(Obstacle((-5, 5), 3.0, 3.0, 0.0, 0.1))

        station = Entity("Station", (0,0), 4.0, 1.0)
        world.add_agent(station)

        path1 = list(zip(np.linspace(9, 7, frames), np.linspace(0, 7, frames)))
        path2 = list(zip(np.linspace(-1, 8, frames), np.linspace(1, 4, frames)))
        path3 = list(zip(np.linspace(-8, -2, frames), np.linspace(8, -2, frames))) 

        world.add_agent(Robot("Robot A", path1, 4.0, 0.7, 'b'))
        world.add_agent(Robot("Robot B", path2, 4.0, 0.7, 'c'))
        world.add_agent(Robot("Robot C", path3, 5.0, 0.7, 'm')) 

        vis = Visualizer(world, fps=10) 
        vis.setup()
        
        html_code = vis.get_animation_html(steps=frames, interval=100)
        components.html(html_code, height=900, scrolling=True)
        plt.close(vis.fig)
        
        st.success("Simulation Complete!")