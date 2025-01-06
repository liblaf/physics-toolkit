import jax
import numpy as np
import pyvista as pv
from jaxtyping import Float, PRNGKeyArray

import fem.element.tetra


def test_volume() -> None:
    rng: PRNGKeyArray = jax.random.key(0)
    points: Float[jax.Array, "4 3"] = jax.random.uniform(rng, (4, 3))
    tetra = pv.UnstructuredGrid(
        [4, 0, 1, 2, 3], [pv.CellType.TETRA], np.asarray(points)
    )
    volume: Float[jax.Array, ""] = fem.element.tetra.volume(points)
    tetra: pv.UnstructuredGrid = tetra.compute_cell_sizes(volume=True)  # pyright: ignore [reportAssignmentType]
    np.testing.assert_allclose(volume, abs(tetra.cell_data["Volume"][0]))
