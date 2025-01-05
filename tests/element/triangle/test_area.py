import jax
import numpy as np
import pyvista as pv
from jaxtyping import Float, PRNGKeyArray

import ptk.element.triangle


def test_area() -> None:
    key: PRNGKeyArray = jax.random.key(0)
    points: Float[jax.Array, "3 2"] = jax.random.uniform(key, (3, 3))
    triangle: pv.PolyData = pv.PolyData.from_regular_faces(
        np.asarray(points), [[0, 1, 2]]
    )
    area: Float[jax.Array, ""] = ptk.element.triangle.area(points)
    triangle: pv.PolyData = triangle.compute_cell_sizes(area=True)  # pyright: ignore [reportAssignmentType]
    np.testing.assert_allclose(area, abs(triangle.cell_data["Area"][0]))
