import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multi_robot import Entity, Robot, Obstacle, World, Visualizer

matplotlib.use('Agg')
plt.rcParams['animation.embed_limit'] = 100.0 

st.set_page_config(layout='wide')
st.title('🤖 Multi-Robot Planning for Communication Coverage Optimization')

# Sidebar
st.sidebar.title('Settings')
resolution = st.sidebar.select_slider('Map Resolution', options=[40, 60, 80, 100], value=60)
frames = st.sidebar.select_slider('Simulation Frames', options=[30, 40, 50, 80], value=40)

if st.button("🚀 Run Simulation", type="primary"):
    
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