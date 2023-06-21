# hyperelastic
Constitutive hyperelastic material formulations for FElupe.

This package provides spaces on which a given material formulation should be projected to, e.g. to the distortional (part of the deformation) space. A generalized framework for isotropic hyperelastic material formulations based on the invariants of the right Cauchy-Green deformation tensor enables a clean coding of these material formulations.

# Example
```python
import hyperelastic as hel

umat = hel.spaces.DistortionalSpace(
    hel.isotropic.invariants.Framework(
        hel.isotropic.invariants.Polynomial(C10=0.5)
    )
)
```
