from dataclasses import dataclass
import math
import casadi as ca


@dataclass
class Position2D:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Position3D(Position2D):
    z: float = 0.0


@dataclass
class BodyVelocity2D:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


@dataclass
class SpatialVelocity2D(BodyVelocity2D):
    pass


@dataclass
class Pose2D(Position2D):
    theta: float = 0.0


@dataclass
class FrenetPosition2D:
    s: float = 0.0
    t: float = 0.0


@dataclass
class FrenetPose2D(FrenetPosition2D):
    xi: float = 0.0


def lateral_sign(position: Position2D, p0: Pose2D):
    return math.copysign(
        1.0,
        (position.y - p0.y) * math.cos(p0.theta)
        - (position.x - p0.x) * math.sin(p0.theta),
    )


def lateral_sign_casadi(position: ca.DM, p0: ca.DM):
    return ca.sign(
        (position[1] - p0[1]) * ca.cos(p0[2]) - (position[0] - p0[0]) * ca.sin(p0[2])
    )


def body2spatial(vb: BodyVelocity2D, p0: Pose2D) -> SpatialVelocity2D:
    return SpatialVelocity2D(
        vx=vb.vx * math.cos(p0.theta) - vb.vy * math.sin(p0.theta),
        vy=vb.vx * math.sin(p0.theta) + vb.vy * math.cos(p0.theta),
        omega=vb.omega,
    )


def spatial2body(vs: SpatialVelocity2D, p0: Pose2D) -> BodyVelocity2D:
    return BodyVelocity2D(
        vx=vs.vx * math.cos(p0.theta) + vs.vy * math.sin(p0.theta),
        vy=-vs.vx * math.sin(p0.theta) + vs.vy * math.cos(p0.theta),
        omega=vs.omega,
    )


def align_yaw(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = ca.atan2(ca.sin(d_yaw), ca.cos(d_yaw))
    return d_yaw_aligned + yaw_2


def align_yaw_function(n=1):
    yaw_1 = ca.MX.sym("yaw_1")
    yaw_2 = ca.MX.sym("yaw_2")
    out = align_yaw(yaw_1, yaw_2)
    return ca.Function("align_yaw", [yaw_1, yaw_2], [out]).map(n)


def align_abscissa(s1, s2, s_total):
    k = ca.fabs(s1 - s2) + s_total / 2.0
    l = k - ca.fmod(ca.fabs(s2 - s1) + s_total / 2.0, s_total)
    return s1 + l * ca.sign(s2 - s1)


def align_abscissa_function(n=1):
    s1 = ca.MX.sym("s1")
    s2 = ca.MX.sym("s2")
    s_total = ca.MX.sym("s_total")
    out = align_abscissa(s1, s2, s_total)
    return ca.Function("align_abscissa", [s1, s2, s_total], [out]).map(n)


def global2frenet(p, p0, yaw):
    cos_theta = ca.cos(yaw)
    sin_theta = ca.sin(yaw)
    R = ca.vertcat(ca.horzcat(cos_theta, -sin_theta), ca.horzcat(sin_theta, cos_theta))
    return R @ (p - p0)


def global2frenet_function(n=1):
    p = ca.MX.sym("p", 2)
    p0 = ca.MX.sym("p0", 2)
    yaw = ca.MX.sym("yaw")
    out = global2frenet(p, p0, yaw)
    return ca.Function("global2frenet", [p, p0, yaw], [out]).map(n)
