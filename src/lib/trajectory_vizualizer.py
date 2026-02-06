import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.quiver import Quiver
from matplotlib.backend_bases import MouseEvent
from typing import Optional
from numpy.typing import NDArray


class TEBVisualizer:
    """Interactive visualizer for Timed Elastic Band (TEB) trajectory optimization.

    Provides a matplotlib-based interface for visualizing robot trajectories with
    interactive obstacle placement. Users can add obstacles with left-click and
    remove them with right-click.

    Attributes:
        fig: Matplotlib figure object.
        ax: Matplotlib axes object for plotting.
        obstacles: List of obstacles as [x, y, radius] arrays.
        obstacle_radius: Fixed radius for all obstacles.
        start_pos: Start position as [x, y, theta] numpy array.
        goal_pos: Goal position as [x, y, theta] numpy array.
        path_line: Line2D object for trajectory visualization.
        start_scatter: PathCollection for start position marker.
        goal_scatter: PathCollection for goal position marker.
        obstacle_patches: List of Circle patches for obstacle visualization.
        quiver: Quiver object for orientation arrows.
        cid: Connection ID for mouse click callback.
    """

    fig: Figure
    ax: Axes
    obstacles: list[list[float]]
    obstacle_radius: float
    start_pos: Optional[NDArray[np.floating]]
    goal_pos: Optional[NDArray[np.floating]]
    path_line: Line2D
    start_scatter: PathCollection
    goal_scatter: PathCollection
    obstacle_patches: list[patches.Circle]
    quiver: Optional[Quiver]
    cid: int

    def __init__(
        self,
        x_lim: tuple[float, float] = (0, 10),
        y_lim: tuple[float, float] = (0, 10),
        title: str = "TEB Optimization",
    ) -> None:
        """Initialize the visualizer with specified plot dimensions.

        Args:
            x_lim: X-axis limits as (min, max) tuple.
            y_lim: Y-axis limits as (min, max) tuple.
            title: Title for the plot window.
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(
            f"{title}\nLeft Click: Add Obstacle | Right Click: Remove Obstacle"
        )
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

        self.obstacles = []
        self.obstacle_radius = 0.5
        self.start_pos = None
        self.goal_pos = None

        (self.path_line,) = self.ax.plot(
            [], [], "b.-", alpha=0.6, linewidth=1, label="Trajectory"
        )
        self.start_scatter = self.ax.scatter(
            [], [], color="green", marker="s", s=100, label="Start", zorder=5
        )
        self.goal_scatter = self.ax.scatter(
            [], [], color="red", marker="*", s=150, label="Goal", zorder=5
        )
        self.obstacle_patches = []
        self.quiver = None

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.ax.legend(loc="upper right")
        plt.ion()
        plt.show()

    def set_start_goal(
        self,
        start: NDArray[np.floating] | list[float],
        goal: NDArray[np.floating] | list[float],
    ) -> None:
        """Set and display start and goal positions.

        Args:
            start: Start position as [x, y, theta] array-like.
            goal: Goal position as [x, y, theta] array-like.
        """
        self.start_pos = np.array(start)
        self.goal_pos = np.array(goal)

        self.start_scatter.set_offsets([self.start_pos[:2]])
        self.goal_scatter.set_offsets([self.goal_pos[:2]])

    def update_trajectory(self, poses: Optional[NDArray[np.floating]]) -> None:
        """Update the displayed trajectory with new poses.

        Args:
            poses: Nx3 numpy array where each row is [x, y, theta].
                  Returns early if None or empty.
        """
        if poses is None or len(poses) == 0:
            return

        self.path_line.set_data(poses[:, 0], poses[:, 1])

        if self.quiver:
            self.quiver.remove()

        u = np.cos(poses[:, 2])
        v = np.sin(poses[:, 2])
        self.quiver = self.ax.quiver(
            poses[:, 0],
            poses[:, 1],
            u,
            v,
            color="blue",
            scale=20,
            width=0.003,
            alpha=0.8,
            zorder=4,
        )

    def get_obstacles(self) -> NDArray[np.floating]:
        """Get all obstacles in the environment.

        Returns:
            Nx3 numpy array where each row is [x, y, radius].
            Returns empty (0, 3) array if no obstacles exist.
        """
        if not self.obstacles:
            return np.empty((0, 3))
        return np.array(self.obstacles)

    def draw(self, pause_time: float = 0.01) -> None:
        """Refresh the plot display.

        Args:
            pause_time: Time in seconds to pause for rendering.
        """
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(pause_time)

    def _on_click(self, event: MouseEvent) -> None:
        """Handle mouse click events for obstacle management.

        Args:
            event: MouseEvent from matplotlib containing click information.
        """
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:
            self.obstacles.append([x, y, self.obstacle_radius])
            self._add_obstacle_patch(x, y, self.obstacle_radius)

        elif event.button == 3:
            self._remove_nearest_obstacle(x, y)

    def _add_obstacle_patch(self, x: float, y: float, r: float) -> None:
        """Create and display a circular obstacle patch.

        Args:
            x: X-coordinate of obstacle center.
            y: Y-coordinate of obstacle center.
            r: Radius of obstacle.
        """
        circle = patches.Circle((x, y), r, color="black", alpha=0.5)
        self.ax.add_patch(circle)
        self.obstacle_patches.append(circle)

    def _remove_nearest_obstacle(self, x: float, y: float) -> None:
        """Remove the nearest obstacle to a clicked position.

        Args:
            x: X-coordinate of click position.
            y: Y-coordinate of click position.
        """
        if not self.obstacles:
            return

        obs_arr = np.array(self.obstacles)
        dists = np.hypot(obs_arr[:, 0] - x, obs_arr[:, 1] - y)
        idx = np.argmin(dists)

        if dists[idx] < 1.0:
            self.obstacles.pop(idx)
            patch = self.obstacle_patches.pop(idx)
            patch.remove()
