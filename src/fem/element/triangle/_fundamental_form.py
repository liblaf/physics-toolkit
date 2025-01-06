import jax
from jaxtyping import Float

import fem.utils


@fem.utils.jit()
def first_fundamental_form(points: Float[jax.Array, "3 3"]) -> Float[jax.Array, "2 2"]:
    r"""First fundamental form.

    $$
    I = \mathrm{d}\bm{r}^T \mathrm{d}\bm{r}
    $$

    References:
        [1] Chen, Zhen, et al. "Fine wrinkling on coarsely meshed thin shells." ACM Transactions on Graphics (TOG) 40.5 (2021): 1-32.
    """
    dr: Float[jax.Array, "3 2"] = (points[1:] - points[0]).T
    I: Float[jax.Array, "2 2"] = dr.T @ dr
    return I
