import itertools

import jax
import numpy as np
from jaxtyping import Float, PRNGKeyArray

import ptk.element.tetra


def test_grad_op() -> None:
    rng: PRNGKeyArray = jax.random.key(0)
    points: Float[jax.Array, "4 3"] = jax.random.uniform(rng, (4, 3))
    f: Float[jax.Array, "4"] = jax.random.uniform(rng, (4,))
    grad_op: Float[jax.Array, "3 4"] = ptk.element.tetra.grad_op(points)
    grad: Float[jax.Array, "3 3"] = grad_op @ f
    for i, j in itertools.combinations(range(4), 2):
        np.testing.assert_allclose(
            grad @ (points[i] - points[j]), f[i] - f[j], rtol=1e-5
        )
