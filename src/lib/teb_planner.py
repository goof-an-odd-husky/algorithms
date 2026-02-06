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
        A_x, A_y = parameters[0]
        B_x, B_y = parameters[1]

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


class SegmentVelocityCost(pyceres.CostFunction):
    def __init__(self, weight: float, max_v: float):
        super().__init__()
        self.weight = weight

        self.max_v = max_v

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        A_x, A_y = parameters[0]
        B_x, B_y = parameters[1]

        s = parameters[2][0]

        AB_x = B_x - A_x
        AB_y = B_y - A_y

        AB_len = np.sqrt(AB_x**2 + AB_y**2 + 1e-10)

        dt_part = np.exp(s)
        t = DT_MIN + dt_part
        v = AB_len / t

        weight_factor = 2.0  # todo: move into constant
        residuals[:] = upper_bound(v, self.max_v, weight_factor)

        if jacobians is not None:
            if v < self.max_v:
                if jacobians[0] is not None:
                    jacobians[0][:] = [0.0, 0.0]
                if jacobians[1] is not None:
                    jacobians[1][:] = [0.0, 0.0]
                if jacobians[2] is not None:
                    jacobians[2][:] = [0.0]
                return True

            # r = w * (v - max)^2 =>
            dr_dv = 2 * weight_factor * (v - self.max_v)

            dr_dv_dv_dl = dr_dv / t

            if jacobians[0] is not None:
                dr_dAx = dr_dv_dv_dl * (-AB_x / AB_len)
                dr_dAy = dr_dv_dv_dl * (-AB_y / AB_len)
                jacobians[0][:] = [dr_dAx, dr_dAy]

            if jacobians[1] is not None:
                dr_dBx = dr_dv_dv_dl * (AB_x / AB_len)
                dr_dBy = dr_dv_dv_dl * (AB_y / AB_len)
                jacobians[1][:] = [dr_dBx, dr_dBy]

            if jacobians[2] is not None:
                dv_ds = -(v / t) * dt_part
                dr_ds = dr_dv * dv_ds
                jacobians[2][:] = [dr_ds]

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

        # pyceres binds to the memory of these specific numpy arrays
        # therefore we have save lists long-term
        self.optimization_xy = []
        self.optimization_theta = []
        self.optimization_dt = []

    @override
    def plan(self) -> NDArray[np.floating] | None:
        if np.array_equal(self.start_pose, self.goal_pose):
            return np.array([np.append(self.start_pose, 0.0)])

        dist = np.linalg.norm(self.goal_pose[:2] - self.start_pose[:2])
        n_points = max(2, int(np.ceil(dist / self.initial_step)) + 1)

        xs = np.linspace(self.start_pose[0], self.goal_pose[0], n_points)
        ys = np.linspace(self.start_pose[1], self.goal_pose[1], n_points)

        delta = self.goal_pose[:2] - self.start_pose[:2]
        goal_heading = np.arctan2(delta[1], delta[0])

        dt_init = self.initial_step / self.max_v

        self.optimization_xy = []
        self.optimization_theta = []
        self.optimization_dt = []

        for i, (x, y) in enumerate(zip(xs, ys)):
            self.optimization_xy.append(np.array([x, y], dtype=np.float64))

            if i == 0:
                th = self.start_pose[2]
            elif i == n_points - 1:
                th = self.goal_pose[2]
            else:
                th = goal_heading
            self.optimization_theta.append(np.array([th], dtype=np.float64))

            if i < n_points - 1:
                self.optimization_dt.append(np.array([dt_init], dtype=np.float64))

        return self.get_trajectory()

    def get_trajectory(self):
        if not self.optimization_xy:
            return np.array([])

        xy = np.array(self.optimization_xy)
        theta = np.array(self.optimization_theta)

        if self.optimization_dt:
            dts = np.array(self.optimization_dt).flatten()
            dts = np.append(dts, 0.0).reshape(-1, 1)
        else:
            dts = np.zeros((len(xy), 1))

        return np.hstack((xy, theta, dts))

    @override
    def refine(self, iterations: int = 1) -> bool:
        if not self.optimization_xy:
            return False

        problem = pyceres.Problem()

        obstacle_cost = SegmentObstaclesCost(
            self.obstacles, weight=10.0, safety_radius=1.0
        )
        velocity_cost = SegmentVelocityCost(
            weight=10.0,
            max_v=self.max_v,
        )

        n_points = len(self.optimization_xy)

        for i in range(n_points - 1):
            xy_curr = self.optimization_xy[i]
            xy_next = self.optimization_xy[i + 1]
            dt = self.optimization_dt[i]

            # theta_curr = self.optimization_theta[i]

            problem.add_residual_block(obstacle_cost, None, [xy_curr, xy_next])

            problem.add_residual_block(velocity_cost, None, [xy_curr, xy_next, dt])

        problem.set_parameter_block_constant(self.optimization_xy[0])
        problem.set_parameter_block_constant(self.optimization_xy[-1])
        # problem.set_parameter_block_constant(self.optimization_theta[0])
        # problem.set_parameter_block_constant(self.optimization_theta[-1])

        options = pyceres.SolverOptions()
        options.max_num_iterations = 50
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        return summary.termination_type == pyceres.TerminationType.CONVERGENCE
