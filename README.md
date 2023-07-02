<p align="center">
  <a href="https://github.com/adtzlr/hyperelastic"><img src="https://github.com/adtzlr/hyperelastic/assets/5793153/499f3f9a-6e1d-4b37-877f-bf8d519e4fe6" height="160px"/></a>
  <p align="center">Constitutive <b>hyperelastic</b> material formulations for <a href="https://github.com/adtzlr/felupe">FElupe</a>.</p>
</p>

[![PyPI version shields.io](https://img.shields.io/pypi/v/hyperelastic.svg)](https://pypi.python.org/pypi/hyperelastic/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black)

This package provides the essential building blocks for constitutive hyperelastic material formulations. This includes material behaviour-independent spaces and frameworks as well as material behaviour-dependent model formulations.

**Spaces** are partial deformations on which a given material formulation should be projected to, e.g. to the distortional (part of the deformation) space. Generalized **Frameworks** for isotropic hyperelastic material formulations based on the invariants of the right Cauchy-Green deformation tensor and the principal stretches enable a clean coding of isotropic material formulations.

The math-module provides helpers in reduced vector ([Voigt](https://en.wikipedia.org/wiki/Voigt_notation)) storage for symmetric three-dimensional second-order tensors along with a matrix storage for (at least minor) symmetric three-dimensional fourth-order tensors. Shear terms are not doubled for strain-like tensors, instead all math operations take care of the reduced vector storage.

$$ \boldsymbol{C} = \begin{bmatrix} C_{11} & C_{22} & C_{33} & C_{12} & C_{23} & C_{13} \end{bmatrix}^T $$

$$ \mathbb{A} = \begin{bmatrix} C_{11} C_{22} C_{33} C_{12} C_{23} C_{13} \end{bmatrix}^T $$

# Installation
Install Python, fire up 🔥 a terminal and run 🏃

```shell
pip install hyperelastic
```

# Usage
Material model formulations have to be created as classes with methods for the evaluation of the `gradient` (stress) and the `hessian` of the strain energy function (elasticity). It depends on the framework which derivatives have to be defined, e.g. the derivatives w.r.t. the invariants of the right Cauchy-Green deformation tensor or w.r.t. the principal stretches.

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


model1 = MyModel1()
umat1 = hel.spaces.DistortionalSpace(hel.frameworks.InvariantsFramework(model1))
```

### Available isotropic hyperelastic invariant-based material formulations
The typical polynomial-based material formulations ([Neo-Hooke](https://en.wikipedia.org/wiki/Neo-Hookean_solid), [Mooney-Rivlin](https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid), [Yeoh](https://en.wikipedia.org/wiki/Yeoh_(hyperelastic_model)) are all available as submodels of the third order deformation material formulation.

- [Third-Order-Deformation (James-Green-Simpson)](https://onlinelibrary.wiley.com/doi/abs/10.1002/app.1975.070190723) ([code](https://github.com/adtzlr/hyperelastic/blob/main/src/hyperelastic/models/invariants/_third_order_deformation.py))


## Principal stretch-based material formulations
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


model2 = MyModel2()
umat2 = hel.spaces.DistortionalSpace(hel.frameworks.StretchesFramework(model2))
```

### Available isotropic hyperelastic stretch-based material formulations
- [Ogden](https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model)) ([code](https://github.com/adtzlr/hyperelastic/blob/main/src/hyperelastic/models/stretches/_ogden.py))

# License
Hyperelastic - Constitutive hyperelastic material formulations for FElupe (C) 2023 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
