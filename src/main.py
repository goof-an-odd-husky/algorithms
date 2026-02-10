from lib.teb_planner import TEBPlanner
from lib.trajectory_vizualizer import TEBVisualizer


def main():
    vizualizer = TEBVisualizer(x_lim=(0, 10), y_lim=(0, 10), path_render_mode="both")

    start = [1.0, 1.0, 0.0]
    goal = [9.0, 9.0, 1.57]
    vizualizer.set_start_goal(start, goal)

    planner = TEBPlanner(start, goal, 1, 1)

    trajectory = planner.plan()

    print("Running... Click on plot to add obstacles. Close window to stop.")

    try:
        while vizualizer.is_open:
            obstacle_data = vizualizer.get_obstacles()

            planner.update_obstacles(obstacle_data)
            planner.refine()
            trajectory = planner.get_trajectory()

            vizualizer.update_trajectory(trajectory)
            vizualizer.draw()

    except KeyboardInterrupt:
        print("Stopped by user (Ctrl+C).")
    finally:
        print("Exited.")


if __name__ == "__main__":
    main()
