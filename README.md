<p align="center">
  <a href="https://github.com/adtzlr/hyperelastic"><img src="https://github.com/adtzlr/hyperelastic/assets/5793153/d875ecd0-a23f-4c11-87c4-0aa99297ab6d" height="160px"/></a>
  <p align="center">Constitutive <b>hyperelastic</b> material formulations for <a href="https://github.com/adtzlr/felupe">FElupe</a>.</p>
</p>

[![PyPI version shields.io](https://img.shields.io/pypi/v/hyperelastic.svg)](https://pypi.python.org/pypi/hyperelastic/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/gh/adtzlr/hyperelastic/branch/main/graph/badge.svg)](https://codecov.io/gh/adtzlr/hyperelastic) [![DOI](https://zenodo.org/badge/656860854.svg)](https://zenodo.org/badge/latestdoi/656860854) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black)

This package provides the essential building blocks for constitutive hyperelastic material formulations. This includes material behaviour-independent spaces and frameworks as well as material behaviour-dependent model formulations.

**Spaces** are full or partial deformations on which a given material formulation should be projected to, e.g. to the distortional (part of the deformation) space. Generalized *Total-Lagrange* **Frameworks** for isotropic hyperelastic material formulations based on the invariants of the right Cauchy-Green deformation tensor and the principal stretches enable a clean coding of isotropic material formulations.

The math-module provides helpers in reduced vector ([Voigt](https://en.wikipedia.org/wiki/Voigt_notation)) storage for symmetric three-dimensional second-order tensors along with a matrix storage for (at least minor) symmetric three-dimensional fourth-order tensors. Shear terms are not doubled for strain-like tensors, instead all math operations take care of the reduced vector storage.

$$ \boldsymbol{C} = \begin{bmatrix} C_{11} & C_{22} & C_{33} & C_{12} & C_{23} & C_{13} \end{bmatrix}^T $$

# Installation
Install Python, fire up 🔥 a terminal and run 🏃

```shell
pip install hyperelastic
```

# Usage
Material model formulations have to be created as classes with methods for the evaluation of the `gradient` (stress) and the `hessian` of the strain energy function (elasticity). It depends on the framework which derivatives have to be defined, e.g. the derivatives w.r.t. the invariants of the right Cauchy-Green deformation tensor or w.r.t. the principal stretches. An instance of a **Framework** has to be *finalized* by the application on a **Space**.

- [Deformation (Full) Space](https://github.com/adtzlr/hyperelastic/blob/main/src/hyperelastic/spaces/_deformation.py)
- [Distortional Space](https://github.com/adtzlr/hyperelastic/blob/main/src/hyperelastic/spaces/_distortional.py)
- [Dilatational Space](https://github.com/adtzlr/hyperelastic/blob/main/src/hyperelastic/spaces/_dilatational.py)

> ⓘ **Note**
> Define your own material model formulation with manual, automatic or symbolic differentiation with the help of your favourite package, e.g. [PyTorch](https://pytorch.org/), [JAX](https://jax.readthedocs.io/en/latest/), [Tensorflow](https://www.tensorflow.org/), [TensorTRAX](https://github.com/adtzlr/tensortrax), [SymPy](https://www.sympy.org/en/index.html), etc.

First, let's import hyperelastic (and its math module).

```python
import hyperelastic as hel
import hyperelastic.math as hm
```

## Invariant-based material formulations
A minimal template for an invariant-based material formulation applied on the distortional space:

```python
class MyInvariantsModel:
    def gradient(self, I1, I2, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the invariants of the right Cauchy-Green deformation tensor."""

        return dWdI1, dWdI2, statevars

    def hessian(self, I1, I2, statevars_old):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the invariants of the right Cauchy-Green deformation tensor."""

        return d2WdI1I1, d2WdI2I2, d2WdI1I2


model = MyInvariantsModel()
framework = hel.frameworks.InvariantsFramework(model)
umat = hel.spaces.DistortionalSpace(framework)
```

### Available isotropic hyperelastic invariant-based material formulations
The typical polynomial-based material formulations ([Neo-Hooke](https://en.wikipedia.org/wiki/Neo-Hookean_solid), [Mooney-Rivlin](https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid), [Yeoh](https://en.wikipedia.org/wiki/Yeoh_hyperelastic_model) are all available as submodels of the third order deformation material formulation.

- [Third-Order-Deformation (James-Green-Simpson)](https://onlinelibrary.wiley.com/doi/abs/10.1002/app.1975.070190723) ([code](https://github.com/adtzlr/hyperelastic/blob/main/src/hyperelastic/models/invariants/_third_order_deformation.py))


## Principal stretch-based material formulations
A minimal template for a principal stretch-based material formulation applied on the distortional space:

```python
class MyStretchesModel:
    def gradient(self, λ, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the principal stretches."""

        return [dWdλ1, dWdλ2, dWdλ3], statevars

    def hessian(self, λ, statevars_old):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the principal stretches."""

        return d2Wdλ1dλ1, d2Wdλ2dλ2, d2Wdλ3dλ3, d2Wdλ1dλ2, d2Wdλ2dλ3, d2Wdλ1dλ3


model = MyStretchesModel()
framework = hel.frameworks.StretchesFramework(model)
umat2 = hel.spaces.DistortionalSpace(framework)
```

### Available isotropic hyperelastic stretch-based material formulations
- [Ogden](https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model)) ([code](https://github.com/adtzlr/hyperelastic/blob/main/src/hyperelastic/models/stretches/_ogden.py))

# License
Hyperelastic - Constitutive hyperelastic material formulations for FElupe (C) 2023 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
