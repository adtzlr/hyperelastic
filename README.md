<p align="center">
  <a href="https://github.com/adtzlr/hyperelastic"><img src="https://github.com/adtzlr/hyperelastic/assets/5793153/499f3f9a-6e1d-4b37-877f-bf8d519e4fe6" height="160px"/></a>
  <p align="center">Constitutive <b>hyperelastic</b> material formulations for <a href="https://github.com/adtzlr/felupe">FElupe</a>.</p>
</p>

This package provides the essential building blocks for constitutive hyperelastic material formulations. This includes material behaviour-independent spaces and frameworks as well as material behaviour-dependent model formulations.

Spaces are partial deformations on which a given material formulation should be projected to, e.g. to the distortional (part of the deformation) space. Generalized frameworks for isotropic hyperelastic material formulations based on the invariants of the right Cauchy-Green deformation tensor and the principal stretches enable a clean coding of isotropic material formulations.

The math module provides helpers in reduced vector (Voigt) storage for symmetric three-dimensional second-order tensors along with a matrix storage for (at least minor) symmetric three-dimensional fourth-order tensors.

# Installation
Install Python, fire up 🔥 a terminal and run 🏃

```shell
pip install hyperelastic
```

# Usage
Material model formulations have to be created as classes with methods for the evaluation of the `gradient` (stress) and the `hessian` (elasticity).

```python
import hyperelastic as hel
import hyperelastic.math as hm
```

## Invariant-based material formulations
A minimal template for an invariant-based material formulation applied on the distortional space:

```python
class MyModel1:
    def gradient(self, I1, I2, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the invariants of the right Cauchy-Green deformation tensor."""

        return dWdI1, dWdI2, statevars

    def hessian(self, I1, I2, statevars_old):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the invariants of the right Cauchy-Green deformation tensor."""

        return d2WdI1I1, d2WdI2I2, d2WdI1I2


umat1 = hel.spaces.DistortionalSpace(hel.frameworks.InvariantsFramework(MyModel1()))
```

## Principal Stretch-based material formulations
A minimal template for a principal stretch-based material formulation applied on the distortional space:

```python
class MyModel2:
    def gradient(self, λ, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the principal stretches."""

        return [dWdλ1, dWdλ2, dWdλ3], statevars

    def hessian(self, λ, statevars_old):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the principal stretches."""

        return d2Wdλ1dλ1, d2Wdλ2dλ2, d2Wdλ3dλ3, d2Wdλ1dλ2, d2Wdλ2dλ3, d2Wdλ1dλ3


umat2 = hel.spaces.DistortionalSpace(hel.frameworks.StretchesFramework(MyModel2()))
```

# License
Hyperelastic - Constitutive hyperelastic material formulations for FElupe (C) 2023 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
