import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.transforms as transforms

from collections import deque


class Entity:
    def __init__(self, name, pos, signal_range, gain, color='r'):
        self.name = name
        self.pos = np.array(pos, dtype=float)
        self.signal_range = signal_range
        self.gain = gain
        self.color = color

    def field(self, X, Y):
        sigma = self.signal_range / 2.0
        distance_sq = (X - self.pos[0])**2 + (Y - self.pos[1])**2
        return self.gain * np.exp(-distance_sq / (2 * sigma**2))


class Robot(Entity):
    def __init__(self, name, path, signal_range, gain, color='b'):
        super().__init__(name, path[0], signal_range, gain, color)
        self.path = path

    def update_position(self, i):
        if i < len(self.path):
            self.pos = np.array(self.path[i])


class Obstacle:
    def __init__(self, center, width, height, angle_deg, attenuation):
        self.center = center
        self.width = width
        self.height = height
        self.angle_deg = angle_deg
        self.attenuation = attenuation

    def get_vertices(self):
        cx, cy = self.center
        w, h, angle_rad = self.width, self.height, np.deg2rad(self.angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        half_w, half_h = w / 2, h / 2
        local_vertices = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        world_vertices = [
            (x * cos_a - y * sin_a + cx, x * sin_a + y * cos_a + cy)
            for x, y in local_vertices
        ]
        return world_vertices

    def create_inside_mask(self, X, Y):
        cx, cy = self.center
        angle_rad = np.deg2rad(self.angle_deg)
        x_t = X - cx
        y_t = Y - cy
        cos_t, sin_t = np.cos(-angle_rad), np.sin(-angle_rad)
        x_local = x_t * cos_t - y_t * sin_t
        y_local = x_t * sin_t + y_t * cos_t
        mask = (np.abs(x_local) < self.width / 2) & (np.abs(y_local) < self.height / 2)
        return mask

    def line_attenuation(self, p1, p2):
        cx, cy = self.center
        angle_rad = np.deg2rad(self.angle_deg)
        cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
        p1_local = np.dot([[cos_a, -sin_a], [sin_a, cos_a]], np.array(p1) - np.array([cx, cy]))
        p2_local = np.dot([[cos_a, -sin_a], [sin_a, cos_a]], np.array(p2) - np.array([cx, cy]))
        if (min(p1_local[0], p2_local[0]) < self.width/2 and
            max(p1_local[0], p2_local[0]) > -self.width/2 and
            min(p1_local[1], p2_local[1]) < self.height/2 and
            max(p1_local[1], p2_local[1]) > -self.height/2):
            return self.attenuation
        return 1.0
    

class World:
    def __init__(self, xlim=(-12,12), ylim=(-12,12), resolution=100):
        self.x_grid = np.linspace(xlim[0], xlim[1], resolution)
        self.y_grid = np.linspace(ylim[0], ylim[1], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.obstacles = []
        self.agents = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_agent(self, agent):
        self.agents.append(agent)

    def link_strength(self, a, b):
        dist = np.linalg.norm(a.pos - b.pos)
        sigma = a.signal_range / 2.0
        s = a.gain * np.exp(-dist**2 / (2 * sigma**2))
        for obs in self.obstacles:
            s *= obs.line_attenuation(a.pos, b.pos)
        return s

    def connected_robots(self, threshold=0.02):
        if not self.agents:
            return set()

        n = len(self.agents)
        adj = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                s_ij = self.link_strength(self.agents[i], self.agents[j])
                s_ji = self.link_strength(self.agents[j], self.agents[i])
                if max(s_ij, s_ji) > threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set([0])
        q = deque([0])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return visited

    def compute_signal_field(self, max_strength=1.0):
        connected = self.connected_robots()
        Z_total = np.zeros_like(self.X)
        for i, agent in enumerate(self.agents):
            if i == 0 or i in connected:
                Z = agent.field(self.X, self.Y)
                for obs in self.obstacles:
                    inside = obs.create_inside_mask(self.X, self.Y)
                    Z[inside] *= obs.attenuation
                Z_total += Z
        return np.clip(Z_total, 0, max_strength)


class Visualizer:
    def __init__(self, world, fps=25):
        self.world = world
        self.fps = fps
        self.fig, self.ax = plt.subplots(figsize=(9,8))
        self.heatmap = None
        self.agent_plots = {}
        self.links = []

    def setup(self):
        ax = self.ax
        ax.set_xlim(-12,12); ax.set_ylim(-12,12)
        ax.set_aspect('equal'); ax.grid(True, linestyle=':')
        for obs in self.world.obstacles:
            rect = patches.Rectangle(
                (-obs.width/2, -obs.height/2),
                obs.width, obs.height,
                facecolor='gray', edgecolor='black', alpha=0.6
            )
            transform = (
                transforms.Affine2D()
                .rotate_deg_around(0,0,obs.angle_deg)
                + transforms.Affine2D().translate(obs.center[0], obs.center[1])
                + ax.transData
            )
            rect.set_transform(transform)
            ax.add_patch(rect)
        Z_init = self.world.compute_signal_field()
        self.heatmap = ax.pcolormesh(
            self.world.X, self.world.Y, Z_init,
            cmap='inferno', vmin=0, vmax=1.0, shading='gouraud'
        )
        for agent in self.world.agents:
            self.agent_plots[agent.name], = ax.plot(
                [], [], marker='o', color=agent.color,
                markersize=8, markeredgecolor='white', label=agent.name
            )
        ax.legend()
        self.fig.colorbar(self.heatmap, ax=ax, label='Signal Strength')

    def animate(self, i):
        for agent in self.world.agents:
            if isinstance(agent, Robot):
                agent.update_position(i)
        Z_total = self.world.compute_signal_field()
        self.heatmap.set_array(Z_total.ravel())

        for agent in self.world.agents:
            self.agent_plots[agent.name].set_data([agent.pos[0]], [agent.pos[1]])

        for line in self.links:
            line.remove()
        self.links.clear()

        connected = self.world.connected_robots()
        for idx_a, a in enumerate(self.world.agents):
            for idx_b, b in enumerate(self.world.agents):
                if idx_b > idx_a and idx_a in connected and idx_b in connected:
                    line, = self.ax.plot(
                        [a.pos[0], b.pos[0]], [a.pos[1], b.pos[1]],
                        color='cyan', alpha=0.5, linewidth=1.5
                    )
                    self.links.append(line)

        return list(self.agent_plots.values()) + [self.heatmap] + self.links

    def run(self, steps, interval=25, save=False):
        ani = animation.FuncAnimation(
            self.fig, self.animate, frames=steps, interval=interval, blit=True
        )
        if save:
            ani.save("./assets/output.gif", writer='pillow', fps=self.fps)
        else:
            plt.show()


if __name__ == "__main__":
    world = World()
    # world.add_obstacle(Obstacle((3, 3), 2, 2, 0, 0.5))

    station = Entity("Station", (0,0), 4.0, 1.0)
    world.add_agent(station)

    frames = 150
    path1 = list(zip(np.linspace(9,7,frames), np.linspace(0,7,frames)))
    path2 = list(zip(np.linspace(-1,8,frames), np.linspace(1,4,frames)))

    world.add_agent(Robot("Robot A", path1, 4.0, 0.4, 'b'))
    world.add_agent(Robot("Robot B", path2, 4.0, 0.4, 'c'))

    vis = Visualizer(world)
    vis.setup()
    vis.run(steps=frames, interval=1, save=True)
