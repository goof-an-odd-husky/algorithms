from lib.teb_planner import TEBPlanner
from lib.trajectory_vizualizer import TEBVisualizer


def main():
    vizualizer = TEBVisualizer(x_lim=(0, 10), y_lim=(0, 10), path_render_mode="both", interactive_obstacles=False)

    start = [1.0, 1.0, 0.0]
    goal = [9.0, 9.0, 1.57]
    vizualizer.set_start_goal(start, goal)

    planner = TEBPlanner(start, goal, 1, 1)

    trajectory = planner.plan()

    print("Running... Click on plot to add obstacles. Close window to stop.")

    try:
        i = 0
        while vizualizer.is_open:
            i += 1
            detected_obstacles = [
                [5.0, 5.0, 1.0],
                [3.0 + i*0.05, 3.0, 0.5],
                [6.0, 2.0]
            ]
            
            vizualizer.set_obstacles(detected_obstacles)

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
