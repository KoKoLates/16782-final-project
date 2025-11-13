import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# functions
def get_signal_field(center_x, center_y, range_std, gain):
    sigma = range_std / 2.0 
    distance_sq = (X - center_x)**2 + (Y - center_y)**2
    return gain * np.exp(-distance_sq / (2 * sigma**2))

def get_obstacle_vertices(center, width, height, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    half_w = width / 2
    half_h = height / 2
    local_vertices = [(-half_w, -half_h), (half_w, -half_h),(half_w, half_h), (-half_w, half_h)]
    world_vertices = []
    for x, y in local_vertices:
        x_rot = x * cos_a - y * sin_a + center[0]
        y_rot = x * sin_a + y * cos_a + center[1]
        world_vertices.append((x_rot, y_rot))
    return world_vertices

def create_inside_obstacle_mask(X, Y, center, width, height, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    x_translated = X - center[0]
    y_translated = Y - center[1]
    cos_theta = np.cos(-angle_rad)
    sin_theta = np.sin(-angle_rad)
    x_local = x_translated * cos_theta - y_translated * sin_theta
    y_local = x_translated * sin_theta + y_translated * cos_theta
    half_w = width / 2.0
    half_h = height / 2.0
    inside_mask = (np.abs(x_local) < half_w) & (np.abs(y_local) < half_h)
    return inside_mask

def create_shadow_mask(X, Y, source_pos, obs_params, attenuation_factor):
    s_x, s_y = source_pos
    o_c, o_w, o_h, o_a = obs_params
    vertices = get_obstacle_vertices(o_c, o_w, o_h, o_a)
    vertex_angles = [np.arctan2(v[1] - s_y, v[0] - s_x) for v in vertices]
    angle_center = np.arctan2(o_c[1] - s_y, o_c[0] - s_x)
    relative_angles = (np.array(vertex_angles) - angle_center + np.pi) % (2 * np.pi) - np.pi
    min_rel_angle = np.min(relative_angles)
    max_rel_angle = np.max(relative_angles)
    shadow_angle_1 = (angle_center + min_rel_angle + np.pi) % (2 * np.pi) - np.pi
    shadow_angle_2 = (angle_center + max_rel_angle + np.pi) % (2 * np.pi) - np.pi
    Grid_Angles = np.arctan2(Y - s_y, X - s_x)
    a1, a2 = shadow_angle_1, shadow_angle_2
    if a1 <= a2:
        is_in_wedge = (Grid_Angles >= a1) & (Grid_Angles <= a2)
    else: 
        is_in_wedge = (Grid_Angles >= a1) | (Grid_Angles <= a2)
    vertex_dists_sq = [(v[0] - s_x)**2 + (v[1] - s_y)**2 for v in vertices]
    min_dist_sq = np.min(vertex_dists_sq)
    Grid_Dist_Sq = (Y - s_y)**2 + (X - s_x)**2
    is_behind = (Grid_Dist_Sq >= min_dist_sq * 0.95)
    is_inside_obstacle = create_inside_obstacle_mask(X, Y, o_c, o_w, o_h, o_a)
    shadow_mask_bool = (is_in_wedge & is_behind) | is_inside_obstacle
    mask = np.ones_like(X)
    mask[shadow_mask_bool] = attenuation_factor
    return mask

def init():
    robot_plot.set_data([], [])
    return robot_plot, heatmap

def animate(i):
    x = x_path[i]
    y = y_path[i]
    robot_pos = (x, y)
    robot_plot.set_data([x], [y])
    
    distance_to_station = math.sqrt((x - start_pos[0])**2 + (y - start_pos[1])**2)
    if distance_to_station > disconnect_threshold:
        Z_robot_attenuated = np.zeros_like(Z_station_base)
    else:
        Z_robot_base = get_signal_field(x, y, signal_range, robot_gain)
        mask_robot = create_shadow_mask(X, Y, robot_pos, obstacle_params, obstacle_attenuation)
        Z_robot_attenuated = Z_robot_base * mask_robot
    
    # overlay
    Z_total = Z_station_attenuated + Z_robot_attenuated
    # saturat
    Z_total = np.clip(Z_total, 0, max_signal_strength)
    # updateeeeeee
    heatmap.set_array(Z_total.ravel())
    
    return robot_plot, heatmap

# Setting
start_pos = (0, 0)
target_pos = (10, 7)
signal_range = 4.0 
disconnect_threshold = 2 * signal_range 

# gain
station_gain = 1.0  
robot_gain = 0.4   
max_signal_strength = 1.0

# obstacle
obstacle_center = (3, 3)
obstacle_width = 4.0
obstacle_height = 1.0
obstacle_angle_deg = 30.0 
obstacle_attenuation = 0.5 

# Animate setting
num_frames = 150  
interval = 40  

# Path
x_path = np.linspace(start_pos[0], target_pos[0], num_frames)
y_path = np.linspace(start_pos[1], target_pos[1], num_frames)

# Map
plot_lim_x = (-12, 12)
plot_lim_y = (-12, 12)
grid_resolution = 300 
x_grid = np.linspace(plot_lim_x[0], plot_lim_x[1], grid_resolution)
y_grid = np.linspace(plot_lim_y[0], plot_lim_y[1], grid_resolution)
X, Y = np.meshgrid(x_grid, y_grid)

# plot
fig, ax = plt.subplots(figsize=(9, 8)) 
ax.set_xlim(plot_lim_x)
ax.set_ylim(plot_lim_y)
ax.set_aspect('equal')
ax.grid(True, linestyle=':')
ax.plot(target_pos[0], target_pos[1], 'gx', markersize=12, mew=3)
ax.plot(x_path, y_path, 'g:', alpha=0.5)
ax.plot(start_pos[0], start_pos[1], 'ro', markersize=10, label='Station') 
rect_patch = patches.Rectangle(
    (-obstacle_width / 2, -obstacle_height / 2), 
    obstacle_width, obstacle_height, 
    facecolor='gray', edgecolor='black', alpha=0.7, label='Obstacle'
)
transform = transforms.Affine2D().rotate_deg_around(0, 0, obstacle_angle_deg) \
    + transforms.Affine2D().translate(obstacle_center[0], obstacle_center[1]) \
    + ax.transData
rect_patch.set_transform(transform)
ax.add_patch(rect_patch)


# set obstacle
obstacle_params = (obstacle_center, obstacle_width, obstacle_height, obstacle_angle_deg)

# cal
Z_station_base = get_signal_field(start_pos[0], start_pos[1], signal_range, station_gain)
mask_station = create_shadow_mask(X, Y, start_pos, obstacle_params, obstacle_attenuation)
Z_station_attenuated = Z_station_base * mask_station

# init set
robot_pos_init = (x_path[0], y_path[0])
Z_robot_base_init = get_signal_field(robot_pos_init[0], robot_pos_init[1], signal_range, robot_gain)
mask_robot_init = create_shadow_mask(X, Y, robot_pos_init, obstacle_params, obstacle_attenuation)
Z_robot_init_attenuated = Z_robot_base_init * mask_robot_init
Z_total_init = Z_station_attenuated + Z_robot_init_attenuated

# saturate
Z_total_init = np.clip(Z_total_init, 0, max_signal_strength)

# pcolormesh
heatmap = ax.pcolormesh(
                X, Y, Z_total_init, 
                cmap='inferno', 
                vmin=0, 
                vmax=max_signal_strength, 
                shading='gouraud', 
                alpha=0.9
            )

# colorbar
cbar = fig.colorbar(heatmap, ax=ax, label='Signal Strength')

# dynamic
robot_plot, = ax.plot([], [], 'bs', markersize=8, label='Robot', markeredgecolor='white')
ax.legend(loc='upper left')

# create
ani = animation.FuncAnimation(fig,
                              animate,
                              init_func=init,
                              frames=num_frames,
                              interval=interval,
                              blit=True)

# save~~~~
# ani.save('Signal_saturated.gif', writer='pillow', fps=1000/interval)
plt.close(fig)
