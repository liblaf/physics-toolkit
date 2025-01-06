import felupe
from felupe import Mesh

import fem


class RegionTetra(fem.region.Region):
    def __init__(
        self,
        mesh: Mesh,
        element: felupe.element.Element | None = None,
        quadrature: felupe.quadrature.Scheme | None = None,
    ) -> None:
        if element is None:
            element = felupe.element.Tetra()
        if quadrature is None:
            quadrature = felupe.quadrature.Tetrahedron(order=1)
        super().__init__(mesh, element, quadrature)
