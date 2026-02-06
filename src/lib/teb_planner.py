from typing import override
from lib.trajectory_planner import TrajectoryPlanner
import numpy as np
from numpy.typing import NDArray
import pyceres


DT_MIN = 0.01


def s_to_dt(s: float) -> float:
    return DT_MIN + np.exp(s)


def upper_bound(value: float, limit: float, weight: float):
    error = value - limit
    if error < 0:
        return 0.0
    return weight * error**2


def lower_bound(value: float, limit: float, weight: float):
    error = limit - value
    if error < 0:
        return 0.0
    return weight * error**2


class SegmentObstaclesCost(pyceres.CostFunction):
    def __init__(
        self, obstacles: NDArray[np.floating], weight: float, safety_radius: float
    ):
        super().__init__()
        self.obstacles_x = obstacles[
            :, 0
        ]  # todo: optimize and use separate arrays from the start?
        self.obstacles_y = obstacles[:, 1]
        self.obstacles_r = obstacles[:, 2]
        self.n_obstacles = len(self.obstacles_r)
        self.weight = weight
        self.safety_radius = safety_radius

        self.set_num_residuals(self.n_obstacles)
        self.set_parameter_block_sizes([2, 2])

    def Evaluate(self, parameters, residuals, jacobians):
        point_A = parameters[0]
        point_B = parameters[1]
        A_x = point_A[0]
        A_y = point_A[1]
        B_x = point_B[0]
        B_y = point_B[1]

        AB_x = B_x - A_x
        AB_y = B_y - A_y
        AB_len_sq = max(AB_x**2 + AB_y**2, 1e-10)

        # O - obstacle
        AO_x = self.obstacles_x - A_x
        AO_y = self.obstacles_y - A_y

        # AO1 - projection of AO onto AB
        # t = |AO1|/|AB|
        t = (AO_x * AB_x + AO_y * AB_y) / AB_len_sq
        t = np.clip(t, 0.0, 1.0)

        O1O_x = AO_x - t * AB_x
        O1O_y = AO_y - t * AB_y

        O1O_len_sq = O1O_x**2 + O1O_y**2
        O1O_len = np.sqrt(O1O_len_sq + 1e-10)

        errors = self.obstacles_r + self.safety_radius - O1O_len
        mask = errors > 0.0
        residuals[:] = np.where(mask, self.weight * errors, 0.0)

        if jacobians is not None:
            # d_hat = O1O / |O1O|
            inv_dist = 1.0 / O1O_len
            d_hat_x = O1O_x * inv_dist
            d_hat_y = O1O_y * inv_dist

            # error < 0 => jacobian = 0
            j_scaler = np.where(mask, self.weight, 0.0)

            if jacobians[0] is not None:
                j_A_factor = j_scaler * (1.0 - t)
                J_A_x = j_A_factor * d_hat_x
                J_A_y = j_A_factor * d_hat_y
                
                jacobians[0][:] = np.vstack((J_A_x, J_A_y)).T.ravel()

            if jacobians[1] is not None:
                j_B_factor = j_scaler * t
                J_B_x = j_B_factor * d_hat_x
                J_B_y = j_B_factor * d_hat_y
                
                jacobians[1][:] = np.vstack((J_B_x, J_B_y)).T.ravel()

        return True


class TEBPlanner(TrajectoryPlanner):
    def __init__(
        self,
        start_pose: NDArray[np.floating] | list[float],
        goal_pose: NDArray[np.floating] | list[float],
        max_v: float,
        max_a: float,
        initial_step: float = 0.5,
    ):
        self.setup_poses(start_pose, goal_pose)
        self.max_v = max_v
        self.max_a = max_a
        self.initial_step = initial_step
        # pyceres needs the same object, so we wrap in our list
        self.optimization_nodes = []

    @override
    def plan(self) -> NDArray[np.floating] | None:
        if np.array_equal(self.start_pose, self.goal_pose):
            return np.array([self.start_pose])

        dist = np.linalg.norm(self.goal_pose[:2] - self.start_pose[:2])
        n_points = max(2, int(np.ceil(dist / self.initial_step)) + 1)

        xs = np.linspace(self.start_pose[0], self.goal_pose[0], n_points)
        ys = np.linspace(self.start_pose[1], self.goal_pose[1], n_points)

        self.optimization_nodes = []
        for x, y in zip(xs, ys):
            self.optimization_nodes.append(np.array([x, y], dtype=np.float64))

        delta = self.goal_pose[:2] - self.start_pose[:2]
        heading = np.arctan2(delta[1], delta[0])
        
        positions = np.array(self.optimization_nodes)
        headings = np.full((len(positions), 1), heading)
        return np.hstack((positions, headings))

    def get_trajectory(self):
        if not self.optimization_nodes:
            return np.array([])
            
        positions = np.array(self.optimization_nodes)
        
        diffs = positions[1:] - positions[:-1]
        diffs = np.vstack((diffs, diffs[-1]))
        
        headings = np.arctan2(diffs[:, 1], diffs[:, 0]).reshape(-1, 1)
        
        return np.hstack((positions, headings))

    @override
    def refine(self, iterations: int = 1) -> bool:
        if not self.optimization_nodes:
            return False
            
        if self.obstacles is None or len(self.obstacles) == 0:
            return True

        problem = pyceres.Problem()

        obstacle_cost = SegmentObstaclesCost(self.obstacles, weight=10.0, safety_radius=1.0)

        for i in range(len(self.optimization_nodes) - 1):
            p_curr = self.optimization_nodes[i]
            p_next = self.optimization_nodes[i+1]

            problem.add_residual_block(obstacle_cost, None, [p_curr, p_next])

        problem.set_parameter_block_constant(self.optimization_nodes[0])
        problem.set_parameter_block_constant(self.optimization_nodes[-1])

        options = pyceres.SolverOptions()
        options.max_num_iterations = 50
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False # True

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)
        
        # print(summary.BriefReport())
        return summary.termination_type == pyceres.TerminationType.CONVERGENCE
