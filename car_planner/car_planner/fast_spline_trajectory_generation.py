import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from .jax_utils import align_yaw_seq


class SplineTrajectory:
    def __init__(self, waypoints, H):
        s = jnp.linspace(0, 1, num=len(waypoints))
        self.spl_x = InterpolatedUnivariateSpline(s, waypoints[:, 0])
        self.spl_y = InterpolatedUnivariateSpline(s, waypoints[:, 1])

        self.intp_s_sym = jnp.linspace(0, 1, num=H * 3)
        self.intp_x = self.spl_x(self.intp_s_sym)
        self.intp_y = self.spl_y(self.intp_s_sym)
        dists = jnp.sqrt(jnp.diff(self.intp_x) ** 2 + jnp.diff(self.intp_y) ** 2)
        self.intp_s = jnp.cumsum(dists)
        self.intp_s -= self.intp_s[0]
        self.intp_yaw = jnp.arctan2(
            self.spl_y.derivative(self.intp_s_sym),
            self.spl_x.derivative(self.intp_s_sym),
        )
        self.intp_yaw = align_yaw_seq(self.intp_yaw)

    @partial(jit, static_argnums=(0,))
    def get_waypoints(self, curr_pos, H, interval):
        nearest_idx = jnp.argmin(
            jnp.sqrt((self.intp_x - curr_pos[0]) ** 2 + (self.intp_y - curr_pos[1]) ** 2)
        )
        s = jnp.linspace(0, interval, num=H) + self.intp_s[nearest_idx]
        x = jnp.interp(s, self.intp_s, self.intp_x)
        y = jnp.interp(s, self.intp_s, self.intp_y)
        yaw = jnp.interp(s, self.intp_s, self.intp_yaw)
        return jnp.stack([x, y, yaw], axis=1)

    # @jit
    # def generate_random_spline_trajectory(H, interval, key):
    #     key, subkey = jax.random.split(key)
    #     key_points = jnp.zeros((3 * 3 + 1, 2))

        
        
@partial(jit, static_argnums=(1,))
def interpolate_action_sequence(action_keypoints, H):
    s = jnp.linspace(0, 1, num=len(action_keypoints))
    intp_s_sym = jnp.linspace(0, 1, num=H)
    f = lambda y: InterpolatedUnivariateSpline(s, y)(intp_s_sym)
    return jax.vmap(f, 1, 1)(action_keypoints)


def test_interpolate_action_sequence(H, key):
    import time
    key, subkey = random.split(key)
    # device_gpu = jax.devices("gpu")[0]
    action_keypoints = random.uniform(subkey, (5, 2))
    # action_keypoints = jax.device_put(action_keypoints, device_gpu)
    start = time.time()
    interpolated_actions = interpolate_action_sequence(action_keypoints, H)
    # import matplotlib.pyplot as plt
    # plt.plot(interpolated_actions[:, 0], interpolated_actions[:, 1])
    # plt.scatter(action_keypoints[:, 0], action_keypoints[:, 1])
    # plt.show()
    return time.time() - start
    
def main():
    key = random.PRNGKey(0)
    test_interpolate_action_sequence(1000, key)
    total_time = 0.0
    for _ in range(100):
        total_time += test_interpolate_action_sequence(1000, key)
    print(f"Average time: {total_time / 100}")
        
    
if __name__ == "__main__":
    main()
    
    