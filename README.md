# hyperelastic
Constitutive hyperelastic material formulations for FElupe.

# Example
```python
import hyperelastic as hel

umat = hel.spaces.DistortionalSpace(
    hel.isotropic.invariants.Framework(
        hel.isotropic.invariants.Polynomial(C10=0.5)
    )
)
```
