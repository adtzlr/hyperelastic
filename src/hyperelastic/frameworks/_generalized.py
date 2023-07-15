from ..models.stretches import GeneralizedInvariantsModel
from ._stretches import Stretches


class GeneralizedInvariants(Stretches):
    def __init__(self, material, fun, nstatevars=0, parallel=False, **kwargs):
        """Initialize the Framework for a generalized invariant-based isotropic
        hyperelastic material formulation."""

        model = GeneralizedInvariantsModel(material, fun, **kwargs)
        super().__init__(material=model, nstatevars=nstatevars, parallel=parallel)
