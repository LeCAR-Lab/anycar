import jax
import jax.numpy as jnp
from jax import jit

@jit
def align_yaw(yaw_1, yaw_2):
    "align yaw_1 onto yaw_2. return the aligned yaw_1."
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw))
    return d_yaw_aligned + yaw_2

@jit
def align_yaw_scan(yaw_1_carry, yaw_2):
    d_yaw = yaw_2 - yaw_1_carry
    d_yaw_aligned = jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw))
    out = d_yaw_aligned + yaw_1_carry
    return out, out

@jit
def align_yaw_seq(yaws):
    return jax.lax.scan(align_yaw_scan, yaws[0], yaws)[1]

def main():
    yaws = jnp.array([-3.12, -3.13, -3.14, 3.13, 3.12])
    yaws_aligned = align_yaw_seq(yaws)
    print(yaws_aligned)

if __name__ == "__main__":
    main()