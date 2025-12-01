import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
from collections import deque

from core import Env
from planner import Path


class ObstacleHandler:
    """Helper class to handle obstacle-based signal attenuation"""
    
    def __init__(self, vertices: List[Tuple[float, float]], attenuation: float = 0.1):
        self.vertices = np.array(vertices)
        self.center = np.mean(self.vertices, axis=0)
        self.attenuation = attenuation
    
    def _is_point_inside(self, p: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = p
        inside = False
        n = len(self.vertices)
        
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            
            if y1 == y2:
                continue
                
            if (y1 > y) != (y2 > y):
                if x < ((x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
        
        return inside
    
    def line_attenuation(self, p1: np.ndarray, p2: np.ndarray, num_samples: int = 10) -> float:
        """Check if line from p1 to p2 intersects obstacle"""
        samples_x = np.linspace(p1[0], p2[0], num_samples)
        samples_y = np.linspace(p1[1], p2[1], num_samples)
        
        for i in range(num_samples):
            p = np.array([samples_x[i], samples_y[i]])
            if self._is_point_inside(p):
                return self.attenuation
        
        return 1.0
    
    def _vectorized_point_in_polygon(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Vectorized point-in-polygon test using winding number"""
        inside = np.zeros_like(X, dtype=bool)
        n = len(self.vertices)
        
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            
            if y1 == y2:
                continue
            
            # Vectorized ray casting
            cond1 = (y1 > Y) != (y2 > Y)
            cond2 = X < ((x2 - x1) * (Y - y1) / (y2 - y1) + x1)
            inside ^= (cond1 & cond2)
        
        return inside
    
    def create_field_attenuation_mask(self, X: np.ndarray, Y: np.ndarray, 
                                     source_pos: np.ndarray) -> np.ndarray:
        """Create shadow mask for field attenuation - OPTIMIZED"""
        s_x, s_y = source_pos
        
        # Calculate angles to vertices
        vertex_angles = np.arctan2(self.vertices[:, 1] - s_y, self.vertices[:, 0] - s_x)
        angle_center = np.arctan2(self.center[1] - s_y, self.center[0] - s_x)
        
        # Find shadow cone angles
        relative_angles = (vertex_angles - angle_center + np.pi) % (2 * np.pi) - np.pi
        min_rel_angle = np.min(relative_angles)
        max_rel_angle = np.max(relative_angles)
        
        shadow_angle_1 = (angle_center + min_rel_angle + np.pi) % (2 * np.pi) - np.pi
        shadow_angle_2 = (angle_center + max_rel_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Grid angles (vectorized)
        Grid_Angles = np.arctan2(Y - s_y, X - s_x)
        
        # Check if in shadow wedge
        a1, a2 = shadow_angle_1, shadow_angle_2
        if a1 <= a2:
            is_in_wedge = (Grid_Angles >= a1) & (Grid_Angles <= a2)
        else:
            is_in_wedge = (Grid_Angles >= a1) | (Grid_Angles <= a2)
        
        # Check if behind obstacle
        vertex_dists_sq = np.sum((self.vertices - source_pos)**2, axis=1)
        min_dist_sq = np.min(vertex_dists_sq)
        Grid_Dist_Sq = (X - s_x)**2 + (Y - s_y)**2
        is_behind = (Grid_Dist_Sq >= min_dist_sq * 0.95)
        
        # Check if inside obstacle (vectorized)
        is_inside = self._vectorized_point_in_polygon(X, Y)
        
        # Combine conditions
        shadow_mask = (is_in_wedge & is_behind) | is_inside
        
        mask = np.ones_like(X, dtype=np.float32)
        mask[shadow_mask] = self.attenuation
        
        return mask


class SignalVisualizer:
    
    def __init__(self, env: Env, resolution: int = 100):  # Reduced from 150
        self.env = env
        self.w, self.h = env.shape
        
        # Create heatmap grid
        xs = np.linspace(0, self.w, resolution)
        ys = np.linspace(0, self.h, resolution)
        self.X, self.Y = np.meshgrid(xs, ys)
        
        # Station parameters (at center)
        self.station_pos = np.array([self.w / 2, self.h / 2])
        self.station_range = 10.0  # How far signal spreads
        self.station_gain = 10.0   # Max brightness at center
        
        # Robot parameters
        self.robot_range = 4.0  # How far signal spreads
        self.robot_gain = 4.0   # Max brightness at center
        
        # Connectivity threshold
        self.connectivity_threshold = 0
        
        # Convert env obstacles to ObstacleHandler objects
        self.obstacle_handlers = [
            ObstacleHandler(obs, attenuation=0.5) 
            for obs in env.obstacles
        ]
        
        # PRE-COMPUTE obstacle masks for inside check (huge speedup!)
        self.obstacle_inside_masks = [
            handler._vectorized_point_in_polygon(self.X, self.Y)
            for handler in self.obstacle_handlers
        ]
        
        # Visualization elements
        self.heatmap = None
        self.robot_plots = []
        self.station_plot = None
        self.link_lines = []
    
    def _gaussian_field(self, src_pos: np.ndarray, signal_range: float, 
                       gain: float) -> np.ndarray:
        """
        Compute Gaussian signal field from a source
        
        signal_range is the radius where signal drops to ~37% (1/e) of max
        """
        sigma = signal_range  # Now signal_range IS the radius
        dx = self.X - src_pos[0]
        dy = self.Y - src_pos[1]
        dist_sq = dx * dx + dy * dy
        return gain * np.exp(-dist_sq / (2 * sigma * sigma))
    
    def _link_strength(self, pos_a: np.ndarray, pos_b: np.ndarray, 
                      range_a: float, gain_a: float) -> float:
        """
        Calculate signal strength from a to b considering obstacles
        
        range_a is the radius where signal drops to ~37% (1/e) of max
        """
        dist = np.linalg.norm(pos_a - pos_b)
        sigma = range_a  # Now range IS the radius
        strength = gain_a * np.exp(-dist**2 / (2 * sigma**2))
        
        # Apply obstacle attenuation
        final_attenuation = 1.0
        for obs_handler in self.obstacle_handlers:
            final_attenuation *= obs_handler.line_attenuation(pos_a, pos_b)
        
        return strength * final_attenuation
    
    def _get_connected_robots(self, robot_positions: List[np.ndarray]) -> set:
        """
        Find which robots are connected to station
        
        Two entities connect if: distance <= (range_A + range_B)
        """
        if not robot_positions:
            return set()
        
        n = len(robot_positions) + 1  # +1 for station
        positions = [self.station_pos] + robot_positions
        ranges = [self.station_range] + [self.robot_range] * len(robot_positions)
        
        # Build adjacency list based on distance threshold
        adj = {i: set() for i in range(n)}
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                max_connection_dist = ranges[i] + ranges[j]
                
                # Check if within connection range (considering obstacles)
                if dist <= max_connection_dist:
                    # Check obstacle attenuation on the link
                    final_attenuation = 1.0
                    for obs_handler in self.obstacle_handlers:
                        final_attenuation *= obs_handler.line_attenuation(
                            positions[i], positions[j]
                        )
                    
                    # Only connect if obstacle attenuation isn't too severe
                    # (attenuation of 0.5 means signal is reduced by half)
                    if final_attenuation > 0.3:  # Threshold for obstacle blocking
                        adj[i].add(j)
                        adj[j].add(i)
        
        # BFS from station (index 0)
        visited = set([0])
        q = deque([0])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        
        # Return robot indices (subtract 1 since station is at index 0)
        return {i - 1 for i in visited if i > 0}
    
    def _compute_signal(self, robot_positions: List[Tuple[float, float]]) -> np.ndarray:
        """Compute total signal field from station and connected robots"""
        robot_pos_arrays = [np.array(p) for p in robot_positions]
        
        # Determine which robots are connected
        connected_robot_indices = self._get_connected_robots(robot_pos_arrays)
        
        Z_total = np.zeros_like(self.X)
        
        # Always add station signal
        Z_station = self._gaussian_field(self.station_pos, self.station_range, 
                                         self.station_gain)
        for obs_handler in self.obstacle_handlers:
            shadow_mask = obs_handler.create_field_attenuation_mask(
                self.X, self.Y, self.station_pos
            )
            Z_station *= shadow_mask
        Z_total += Z_station
        
        # Add signals from connected robots only
        for idx, pos in enumerate(robot_pos_arrays):
            if idx in connected_robot_indices:
                Z_robot = self._gaussian_field(pos, self.robot_range, self.robot_gain)
                for obs_handler in self.obstacle_handlers:
                    shadow_mask = obs_handler.create_field_attenuation_mask(
                        self.X, self.Y, pos
                    )
                    Z_robot *= shadow_mask
                Z_total += Z_robot
        
        # Max possible: station(10) + multiple robots(4 each) could exceed 20
        # But we clip at a reasonable max for visualization
        return np.clip(Z_total, 0, 10.0)
    
    def _prepare_ax(self):
        """Setup the plot axes"""
        fig, ax = plt.subplots(figsize=(9, 8))
        
        # Draw obstacles
        for poly in self.env.obstacles:
            ax.add_patch(Polygon(poly, closed=True, facecolor='gray', alpha=0.6, zorder=8))
        
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect("equal")
        ax.grid(True, color='gray', linewidth=0.5, alpha=0.3)
        
        # Initial heatmap with no robots
        Z_init = self._compute_signal([])
        self.heatmap = ax.pcolormesh(
            self.X, self.Y, Z_init,
            cmap='inferno', vmin=0, vmax=10.0, shading='gouraud'
        )
        
        # Station marker
        self.station_plot = ax.scatter(
            self.station_pos[0], self.station_pos[1],
            marker='x', color='black', s=150, linewidths=2, zorder=10
        )
        
        return fig, ax
    
    def animate(self, paths: List[Path], file_name: Optional[str] = None, 
               interval: int = 40, show_robots: bool = True, 
               show_connections: bool = False):
        fig, ax = self._prepare_ax()
        
        # Robot markers
        if show_robots:
            robot_colors = cm.get_cmap("tab10", len(paths))
            for i in range(len(paths)):
                plot, = ax.plot([], [], marker='o', markersize=8,
                              color=robot_colors(i), label=f'Robot {i+1}')
                self.robot_plots.append(plot)
        
        # ax.legend(loc='lower left')
        fig.colorbar(self.heatmap, ax=ax, label="Signal Strength")
        
        # Number of frames = longest path
        T = max(len(p) for p in paths)
        
        def update(t: int):
            # Get current robot positions
            robot_positions = []
            for i, p in enumerate(paths):
                if t < len(p):
                    state = p[t]
                else:
                    state = p[-1]  # Stay at last position
                
                pos = (state.x, state.y)
                robot_positions.append(pos)
                
                # Update robot marker
                if show_robots and i < len(self.robot_plots):
                    self.robot_plots[i].set_data([pos[0]], [pos[1]])
            
            # Update heatmap
            Z = self._compute_signal(robot_positions)
            self.heatmap.set_array(Z.ravel())
            
            # Update connection lines
            if show_connections:
                # Remove old lines
                for line in self.link_lines:
                    line.remove()
                self.link_lines.clear()
                
                # Draw new connections
                robot_pos_arrays = [np.array(p) for p in robot_positions]
                connected = self._get_connected_robots(robot_pos_arrays)
                
                # Draw lines between connected robots
                for idx_a in connected:
                    for idx_b in connected:
                        if idx_b > idx_a:
                            pos_a = robot_positions[idx_a]
                            pos_b = robot_positions[idx_b]
                            line, = ax.plot(
                                [pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]],
                                color='cyan', alpha=0.4, linewidth=1.5, zorder=5
                            )
                            self.link_lines.append(line)
                
                # Draw lines from station to connected robots
                for idx in connected:
                    pos = robot_positions[idx]
                    line, = ax.plot(
                        [self.station_pos[0], pos[0]], 
                        [self.station_pos[1], pos[1]],
                        color='yellow', alpha=0.3, linewidth=1.5, zorder=5
                    )
                    self.link_lines.append(line)
            
            return self.robot_plots + [self.heatmap, self.station_plot] + self.link_lines
        
        ani = FuncAnimation(
            fig, update,
            frames=T,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        if file_name is not None:
            ani.save(file_name, writer='pillow', fps=int(1000/interval), dpi=150)
            print(f"Animation saved to {file_name}")
        
        plt.show()