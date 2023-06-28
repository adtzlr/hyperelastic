<p align="center">
  <a href="https://github.com/adtzlr/hyperelastic"><img src="https://github.com/adtzlr/hyperelastic/assets/5793153/499f3f9a-6e1d-4b37-877f-bf8d519e4fe6" height="160px"/></a>
  <p align="center">Constitutive <b>hyperelastic</b> material formulations for <a href="https://github.com/adtzlr/felupe">FElupe</a>.</p>
</p>

This package provides the essential building blocks for constitutive hyperelastic material formulations. This includes material behaviour-independent spaces and frameworks as well as material behaviour-dependent model formulations.

Spaces are partial deformations on which a given material formulation should be projected to, e.g. to the distortional (part of the deformation) space. Generalized frameworks for isotropic hyperelastic material formulations based on the invariants of the right Cauchy-Green deformation tensor and the principal stretches enable a clean coding of isotropic material formulations.

The math module provides helpers in reduced vector (Voigt) storage for symmetric three-dimensional second-order tensors along with a matrix storage for (at least minor) symmetric three-dimensional fourth-order tensors.

# Example
Material Formulations have to be implemented as classes with `gradient` (stress) and `hessian` (elasticity) methods.

```python
import hyperelastic as hel
import hyperelastic.math as hm

class MyMaterialModel:
    def __init__(self, shear_modulus):
        self.shear_modulus = shear_modulus

    def gradient(self, I1, I2, statevars_old):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the invariants of the right Cauchy-Green deformation tensor."""

        dWdI1 = self.shear_modulus / 2
        dWdI2 = 0

        # update the state variables
        statevars_new = statevars_old

        return dWdI1, dWdI2, statevars_new

    def hessian(self, I1, I2, statevars_old):
        """The hessian as the second partial derivatives of the strain energy function 
        w.r.t. the invariants of the right Cauchy-Green deformation tensor. """

        d2WdI1I1 = 0
        d2WdI1I2 = 0
        d2WdI2I2 = 0

        return d2WdI1I1, d2WdI2I2, d2WdI1I2


umat = hel.spaces.DistortionalSpace(
    hel.isotropic.FrameworkInvariants(
        MyMaterialModel(shear_modulus=1.0)
    )
)
```
