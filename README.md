<p align="center">
  <a href="https://github.com/adtzlr/hyperelastic"><img src="https://github.com/adtzlr/hyperelastic/assets/5793153/499f3f9a-6e1d-4b37-877f-bf8d519e4fe6" height="160px"/></a>
  <p align="center">Constitutive <b>hyperelastic</b> material formulations for <a href="https://github.com/adtzlr/felupe">FElupe</a>.</p>
</p>

This package provides spaces on which a given material formulation should be projected to, e.g. to the distortional (part of the deformation) space. A generalized framework for isotropic hyperelastic material formulations based on the invariants of the right Cauchy-Green deformation tensor enables a clean coding of these material formulations.

# Example
```python
import hyperelastic as hel

class NeoHooke:
    def __init__(self, C10=0):
        self.C10 = C10

    def gradient(self, x):
        I1, I2, statevars = x

        dWdI1 = self.C10
        dWdI2 = 0

        return [dWdI1, dWdI2, statevars]

    def hessian(self, x):
        I1, I2, statevars = x

        d2WdI1I1 = 0
        d2WdI1I2 = 0
        d2WdI2I2 = 0

        return [d2WdI1I1, d2WdI1I2, d2WdI2I2]


umat = hel.spaces.DistortionalSpace(
    hel.isotropic.invariants.Framework(
        NeoHooke(C10=0.5)
    )
)
```
