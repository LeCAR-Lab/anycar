import os
from car_planner import CAR_PLANNER_ASSETS_DIR
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import casadi as ca
from car_planner.utils import (
    FrenetPose2D,
    Pose2D,
    align_abscissa,
    align_yaw,
    lateral_sign,
    lateral_sign_casadi,
)
import matplotlib.pyplot as plt
import copy
import time


class GlobalTrajectory:
    PX = 0
    PY = 1
    PZ = 2
    YAW = 3
    SPEED = 4
    CURVATURE = 5
    DIST_TO_SF_BWD = 6
    DIST_TO_SF_FWD = 7
    REGION = 8
    LEFT_BOUND_X = 9
    LEFT_BOUND_Y = 10
    RIGHT_BOUND_X = 11
    RIGHT_BOUND_Y = 12
    BANK = 13
    LON_ACC = 14
    LAT_ACC = 15
    YAW_RATE = 16
    VY = 17

    def __init__(self, waypoints: np.ndarray):
        """_summary_
        Args:
            waypoints (np.ndarray): num_pt * x
        """
        self.waypoints = waypoints
        self.kdtree = KDTree(waypoints[:, [self.PX, self.PY]], copy_data=True)

        # append zeros for missing columns
        if waypoints.shape[1] < 18:
            waypoints = np.hstack(
                [
                    waypoints,
                    np.zeros((waypoints.shape[0], 18 - waypoints.shape[1])),
                ]
            )
            waypoints[:, self.SPEED] = 2.0 #10.0

        # make a temp trajectory
        interpolants_original = waypoints[:, [self.PX, self.PY]]
        abscissa_original = ca.linspace(0, 1, interpolants_original.shape[0] + 1)[:-1]
        interpolants = ca.vertcat(
            interpolants_original, interpolants_original, interpolants_original
        )
        abscissa = ca.vertcat(
            abscissa_original - 1, abscissa_original, abscissa_original + 1
        )

        # intp
        x_intp = ca.interpolant(
            "x_intp",
            "bspline",
            abscissa.full().T.tolist(),
            interpolants[:, self.PX].full().squeeze().tolist(),
            {},
        )
        y_intp = ca.interpolant(
            "y_intp",
            "bspline",
            abscissa.full().T.tolist(),
            interpolants[:, self.PY].full().squeeze().tolist(),
            {},
        )

        # integrate and get total length
        abscissa_sym = ca.MX.sym("abscissa_sym")
        p_sym = ca.MX.sym("p_sym")
        dx = ca.jacobian(x_intp(abscissa_sym + p_sym), abscissa_sym)
        dy = ca.jacobian(y_intp(abscissa_sym + p_sym), abscissa_sym)
        ds = ca.sqrt(dx**2 + dy**2)
        
        integrator = ca.integrator(
            "integrator",
            "rk",
            {"x": abscissa_sym, "p": p_sym, "ode": ds},
            0.0,
            float(abscissa_original[1]),
        ).map(abscissa_original.size1())
        s = integrator(x0=0, p=abscissa_original)["xf"]
        self.total_length_ = ca.sum2(s)
        s = ca.cumsum(s)
        s -= s[0]
        self.abscissa_ = s

        # now the real trajectory
        self.waypoints = waypoints.T
        interpolants = ca.horzcat(self.waypoints, self.waypoints[:, :4])
        abscissa = ca.horzcat(s, s[:, :4] + self.total_length_)
        interpolants = ca.horzcat(interpolants[:, -7:-4], interpolants)
        abscissa = ca.horzcat(abscissa[:, -7:-4] - self.total_length_, abscissa)

        # TODO: add left and right boundary interpolation
        s_sym = ca.MX.sym("s_sym")
        s_mod = align_abscissa(s_sym, self.total_length_ / 2.0, self.total_length_)
        s_mod_sym = ca.MX.sym("s_mod_sym")
        self.x_intp_impl_ = ca.interpolant(
            "x_intp",
            "bspline",
            abscissa.full().tolist(),
            interpolants[self.PX, :].full().squeeze().tolist(),
            {},
        )
        self.y_intp_impl_ = ca.interpolant(
            "y_intp",
            "bspline",
            abscissa.full().tolist(),
            interpolants[self.PY, :].full().squeeze().tolist(),
            {},
        )
        self.speed_intp_impl_ = ca.interpolant(
            "speed_intp",
            "bspline",
            abscissa.full().tolist(),
            interpolants[self.SPEED, :].full().squeeze().tolist(),
            {},
        )

        # spline yaw
        d2x, dx = ca.hessian(self.x_intp_impl_(s_mod_sym), s_mod_sym)
        d2y, dy = ca.hessian(self.y_intp_impl_(s_mod_sym), s_mod_sym)
        d2x_func = ca.Function("d2x_func", [s_mod_sym], [d2x])
        d2y_func = ca.Function("d2y_func", [s_mod_sym], [d2y])
        dx_func = ca.Function("dx_func", [s_mod_sym], [dx])
        dy_func = ca.Function("dy_func", [s_mod_sym], [dy])
        spline_yaw = ca.atan2(dy_func(s_mod), dx_func(s_mod))
        curvature = dx_func(s_mod) * d2y_func(s_mod) - dy_func(s_mod) * d2x_func(
            s_mod
        ) / ca.sqrt(
            ca.power(ca.power(dx_func(s_mod), 2) + ca.power(dy_func(s_mod), 2), 3)
        )

        self.spline_yaw_intp_ = ca.Function("spline_yaw_intp", [s_sym], [spline_yaw])
        self.curvature_intp_ = ca.Function("curvature_intp", [s_sym], [curvature])
        self.x_intp_ = ca.Function("x_intp", [s_sym], [self.x_intp_impl_(s_mod)])
        self.y_intp_ = ca.Function("y_intp", [s_sym], [self.y_intp_impl_(s_mod)])
        self.speed_intp_ = ca.Function("speed_intp", [s_sym], [self.speed_intp_impl_(s_mod)])

        # yaw intp
        if self.waypoints.shape[0] > 4:
            yaws = interpolants[self.YAW, :]
            for i in range(1, yaws.size1()):
                yaws[i] = align_yaw(yaws[i], yaws[i - 1])
            self.yaw_intp_impl_ = ca.interpolant(
                "yaw_intp", "bspline", abscissa.full().tolist(), yaws.full().squeeze().tolist(), {}
            )
            self.yaw_intp_ = ca.Function(
                "yaw_intp", [s_sym], [self.yaw_intp_impl_(s_mod)]
            )
        else:
            self.yaw_intp_ = None

        # build frenet 2 global function
        t_sym = ca.MX.sym("t_sym")
        xi_sym = ca.MX.sym("xi_sym")
        x0 = self.x_intp_(s_mod)
        y0 = self.y_intp_(s_mod)
        yaw0 = self.spline_yaw_intp_(s_mod)
        d_x = -1.0 * ca.sin(yaw0) * t_sym
        d_y = ca.cos(yaw0) * t_sym
        theta = align_yaw(yaw0 + xi_sym, 0.0)
        out = ca.vertcat(x0 + d_x, y0 + d_y, theta)
        self.frenet_to_global_ = ca.Function(
            "frenet_to_global", [s_sym, t_sym, xi_sym], [out]
        )

        # build global 2 frenet function
        x_sym = ca.MX.sym("x_sym")
        y_sym = ca.MX.sym("y_sym")
        theta_sym = ca.MX.sym("theta_sym")
        s0_sym = ca.MX.sym("s0_sym")
        s0_mod = align_abscissa(s0_sym, self.total_length_ / 2.0, self.total_length_)
        t0_sym = ca.MX.sym("t0_sym")
        dist_sq = ca.sumsqr(
            self.frenet_to_global_(s, 0.0, 0.0)[:2] - ca.vertcat(x_sym, y_sym)
        )
        qp = {"x": s_sym, "f": dist_sq, "p": ca.vertcat(x_sym, y_sym)}
        qp_opts = {
            "jit": True,
            "print_iteration": False,
            "print_status": False,
            "print_time": False,
            "print_header": False,
            "qpsol": "qrqp",
            "qpsol_options": {
                "print_header": False,
                "print_iter": False,
                "print_info": False,
            },
        }
        self.global_to_frenet_sol_ = ca.nlpsol(
            "global_to_frenet", "sqpmethod", qp, qp_opts
        )
        sol = self.global_to_frenet_sol_(x0=s0_mod, p=ca.vertcat(x_sym, y_sym))["x"]
        sol = align_abscissa(sol, self.total_length_ / 2.0, self.total_length_)
        x_out = self.x_intp_(sol)
        y_out = self.y_intp_(sol)
        yaw_out = self.spline_yaw_intp_(sol)
        t_out = ca.hypot(x_out - x_sym, y_out - y_sym) * lateral_sign_casadi(
            ca.vertcat(x_sym, y_sym), ca.vertcat(x_out, y_out, yaw_out)
        )
        xi_out = align_yaw(theta_sym, yaw_out) - yaw_out
        self.global_to_frenet_ = ca.Function(
            "global_to_frenet",
            [ca.vertcat(x_sym, y_sym, theta_sym, s0_sym, t0_sym)],
            [sol, t_out, xi_out],
        )

    def frenet_to_global(self, frenet_pose: FrenetPose2D) -> Pose2D:
        out = self.frenet_to_global_(frenet_pose.s, frenet_pose.t, frenet_pose.xi)
        return Pose2D(x=float(out[0]), y=float(out[1]), theta=float(out[2]))

    def global_to_frenet(self, global_pose: Pose2D) -> FrenetPose2D:
        p0 = FrenetPose2D()
        _, idx = self.kdtree.query([global_pose.x, global_pose.y])
        p0.s = self.abscissa_[idx]

        p0_g = Pose2D()
        p0_g.x = self.waypoints[self.PX, idx]
        p0_g.y = self.waypoints[self.PY, idx]
        p0_g.theta = self.spline_yaw_intp_(p0.s)
        p0.t = ca.hypot(global_pose.x - p0_g.x, global_pose.y - p0_g.y) * lateral_sign(
            global_pose, p0_g
        )
        p0.theta = align_yaw(global_pose.theta, p0_g.theta)

        out = self.global_to_frenet_(
            ca.vertcat(global_pose.x, global_pose.y, global_pose.theta, p0.s, p0.t)
        )
        p0.s = float(out[0])
        p0.t = float(out[1])
        p0.xi = float(out[2])
        return p0

    def get_frenet_to_global_function(self):
        return self.frenet_to_global_

    def get_global_to_frenet_function(self):
        return self.global_to_frenet_

    def get_curvature_intp_function(self):
        return self.curvature_intp_

    def get_left_bound_intp_function(self):
        pass

    def get_right_bound_intp_function(self):
        pass

    def get_x_intp_function(self):
        return self.x_intp_

    def get_y_intp_function(self):
        return self.y_intp_

    def get_yaw_intp_function(self):
        return self.yaw_intp_

    def get_spline_yaw_intp_function(self):
        return self.spline_yaw_intp_

    def get_vx_intp_function(self):
        pass

    def get_vy_intp_function(self):
        pass

    def get_vyaw_intp_function(self):
        pass

    def get_total_length(self):
        return self.total_length_

    def generate(self, obs: np.ndarray, dt: float, h: int, return_frenet_pose = False) -> np.ndarray:
        # TODO: accept a list of vels
        st = time.time()
        pos2d = obs[:2]
        psi = obs[2]
        vel2d = obs[3:5]
        vel = np.linalg.norm(vel2d)
        vel = np.clip(vel, 0.2, float("inf"))
        pose = Pose2D(x=pos2d[0], y=pos2d[1], theta=psi)
        p0 = self.global_to_frenet(pose)

        # s_begin = p0.s
        # s_end = s_begin + vel * dt * h
        # s = np.linspace(s_begin, s_end, h, endpoint=False)
        target_vel = 2.0
        # create linear interpolation for velocity
        vel_intp = np.zeros(h)
        s = np.zeros(h)
        s[0] = p0.s
        vel_intp[0] = self.speed_intp_(s[0])
        for i in range(1, h):
            s[i] = s[i - 1] + vel_intp[i - 1] * dt
            vel_intp[i] = self.speed_intp_(s[i])
        x = self.get_x_intp_function()(s)
        y = self.get_y_intp_function()(s)
        yaw = self.get_spline_yaw_intp_function()(s)
        traj = np.stack([x, y, yaw, vel_intp[:, np.newaxis]], axis=1)[:, :, 0]
        if return_frenet_pose:
            return traj, p0
        return traj


def generate_circle_trajectory(center, radius):
    num_points = 100
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    waypoints = np.stack([x, y], axis=1)
    return waypoints


def generate_oval_trajectory(center, x_radius, y_radius, direction, endpoint=False, rotate_deg=0.0):
    assert direction in [1, -1]
    num_points = 1000
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=endpoint)
    if direction == -1:
        t = np.flip(t)
    x = center[0] + x_radius * np.cos(t)
    y = center[1] + y_radius * np.sin(t)
    waypoints = np.stack([x, y], axis=1)
    
    # rotate the oval
    if rotate_deg != 0.0:
        theta = np.deg2rad(rotate_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        waypoints = np.dot(R, waypoints.T).T
    
    return waypoints

def generate_raceline_trajectory(*args):
    # d = pd.read_csv(os.path.join(CAR_PLANNER_ASSETS_DIR, 'raceline_tum.csv')).values
    # d = pd.read_csv(os.path.join(CAR_PLANNER_ASSETS_DIR, 'dvij.csv')).values
    d = pd.read_csv(os.path.join(CAR_PLANNER_ASSETS_DIR, 'track.csv')).values 
    # d = pd.read_csv(os.path.join(CAR_PLANNER_ASSETS_DIR, 'track_10.csv')).values 
    waypoints = np.array(d)
    # waypoints = (np.array(d) - np.mean(np.array(d), axis = 0)) * 0.2
    # plt.plot(np.array(d)[:, 0], np.array(d)[:, 1])
    # plt.plot(waypoints[:, 0], waypoints[:, 1])
    # plt.show()
    return waypoints
    
def generate_hulu_trajectory(center, x_radius, y_radius, direction):
    assert direction in [1, -1]
    pass

def generate_rectangle_trajectory(center, width, height):
    # note: make sure there are no overlapping points at the corners    
    side_1_x = np.linspace(center[0] - width / 2, center[0] + width / 2, 100)
    side_1_y = np.ones(100) * (center[1] - height / 2)
    side_2_x = np.ones(100) * (center[0] + width / 2)
    side_2_y = np.linspace(center[1] - height / 2, center[1] + height / 2, 100)
    side_3_x = np.linspace(center[0] + width / 2, center[0] - width / 2, 100)
    side_3_y = np.ones(100) * (center[1] + height / 2)
    side_4_x = np.ones(100) * (center[0] - width / 2)
    side_4_y = np.linspace(center[1] + height / 2, center[1] - height / 2, 100)
    x = np.concatenate([side_1_x, side_2_x, side_3_x, side_4_x])
    y = np.concatenate([side_1_y, side_2_y, side_3_y, side_4_y])
    waypoints = np.stack([x, y], axis=1)
    return waypoints


def test_trajectory():
    waypoints = generate_circle_trajectory()
    import time

    t0 = time.time()
    traj = GlobalTrajectory(waypoints)
    t1 = time.time()
    print("Time taken to construct the trajectory: ", t1 - t0)
    total_length = traj.get_total_length()
    print(total_length)
    actual_total_length = 2 * np.pi
    assert np.isclose(total_length, actual_total_length, atol=1e-6)

    s_test = np.linspace(0, np.pi, 100)
    x_test = traj.get_x_intp_function()(s_test)
    y_test = traj.get_y_intp_function()(s_test)
    plt.plot(x_test, y_test)

    # test frenet to global
    p0 = FrenetPose2D()
    p0.s = 1.0
    p0.t = 0.1
    p0.xi = 0.0
    time_start = time.time()
    global_pose = traj.frenet_to_global(p0)
    time_end = time.time()
    print("Time taken to convert frenet to global: ", time_end - time_start)
    # plot the global pose as an arrow
    plt.plot(global_pose.x, global_pose.y, "ro")
    plt.quiver(
        global_pose.x,
        global_pose.y,
        np.cos(global_pose.theta),
        np.sin(global_pose.theta),
    )

    # test global to frenet
    time_start = time.time()
    p1 = traj.global_to_frenet(global_pose)
    time_end = time.time()
    print("Time taken to convert global to frenet: ", time_end - time_start)
    assert np.isclose(p0.s, p1.s, atol=1e-2)
    assert np.isclose(p0.t, p1.t, atol=1e-2)
    assert np.isclose(p0.xi, p1.xi, atol=1e-2)
    # get the projected frenet pose
    p1_proj = copy.deepcopy(p1)
    p1_proj.t = 0.0
    global_pose_proj = traj.frenet_to_global(p1_proj)
    plt.plot(global_pose_proj.x, global_pose_proj.y, "go")
    plt.quiver(
        global_pose_proj.x,
        global_pose_proj.y,
        np.cos(global_pose_proj.theta),
        np.sin(global_pose_proj.theta),
    )

    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    track = generate_oval_trajectory((0., 0.), 1.2, 1.4, 1, endpoint=True)
    import ipdb; ipdb.set_trace()
    
    plt.figure()
    plt.plot(track[:, 0], track[:, 1])
    plt.show()
