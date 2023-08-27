import numpy as np


class GeneralizedInvariantsModel:
    r"""Generalized-invariants isotropic hyperelastic material formulation based on the
    principal stretches.

    ..  math::

        \psi = \psi \left(
            I_1\left( E_1, E_2, E_3 \right),
            I_2\left( E_1, E_2, E_3 \right),
            I_3\left( E_1, E_2, E_3 \right) \right)

    The three principal invariants

    ..  math::

        J_1 &= E_1 + E_2 + E_3

        J_2 &= E_1 E_2 + E_2 E_3 + E_1 E_3

        J_3 &= E_1 E_2 E_3

    are formulated on a one-dimensional strain-stretch relation.

    ..  math::

        E_\alpha &= f(\lambda_\alpha)

        E'_\alpha &= f'(\lambda_\alpha) = \frac{\partial f(\lambda_\alpha)}
            {\partial \lambda_\alpha}

        E''_\alpha &= f''(\lambda_\alpha) = \frac{\partial^2 f(\lambda_\alpha)}
            {\partial \lambda_\alpha~\partial \lambda_\alpha}

    Depending on the strain-stretch relation, the invariants contain deformation-
    independent values.

    ..  math::

        J_{1,0} &= J_1(E_\alpha(\lambda_\alpha=1))

        J_{2,0} &= J_2(E_\alpha(\lambda_\alpha=1))

        J_{3,0} &= J_3(E_\alpha(\lambda_\alpha=1))

    The deformation-dependent parts of the invariants are scaled by deformation-
    independent coefficients of normalization. The deformation-independent parts are
    re-added after the scaling.

    ..  math::

        I_1 &= c_1 (J_1 - J_{1,0}) + J_{1,0}

        I_2 &= c_2 (J_2 - J_{2,0}) + J_{2,0}

        I_3 &= J_3

    Note that the scaling is only applied to the first and second invariant, as the
    third invariant does not contribute to the strain energy function at the undeformed
    state.

    ..  math::

        E_0 &= E(\lambda=1)

        E'_0 &= E'(\lambda=1)

        E''_0 &= E''(\lambda=1)

    The second partial derivative of the strain w.r.t. the stretch must be
    provided for a reference strain, e.g. the Green-Lagrange strain measure (at the
    undeformed state).

    ..  math::

        J''_{1,0} &= \frac{3}{2} \left( E''_0 + E'_0 \right)

        J''_{2,0} &= \frac{3}{2} \left( (2 E_0 (E''_0 + E'_0)) - E'^2_0 \right)

    ..  math::

        c_1 &= \frac{J''_{1,0,ref}}{J''_{1,0}}

        c_2 &= \frac{J''_{2,0,ref}}{J''_{2,0}}

    The first partial derivatives of the strain energy function w.r.t. the invariants

    ..  math::

        \psi_{,1} &= \frac{\partial \psi}{\partial I_1}

        \psi_{,2} &= \frac{\partial \psi}{\partial I_2}

        \psi_{,3} &= \frac{\partial \psi}{\partial I_3}

    and the partial derivatives of the invariants w.r.t. the principal stretches are
    defined. From here on, this is consistent with any invariant-based hyperelastic
    material formulation, except for the factors of normalization.

    ..  math::

        \frac{\partial I_1}{\partial E_\alpha} &= c_1

        \frac{\partial I_2}{\partial E_\alpha} &= c_2 \left( E_\beta + E_\gamma \right)

        \frac{\partial I_3}{\partial E_\alpha} &= E_\beta E_\gamma

    The first partial derivatives of the strain energy density w.r.t. the
    principal stretches are required for the principal values of the stress.

    ..  math::

        \frac{\partial \psi}{\partial \lambda_\alpha} =
            \frac{\partial \psi}{\partial I_1} \frac{\partial I_1}{\partial E_\alpha}
            \frac{\partial E_\alpha}{\partial \lambda_\alpha}
            + \frac{\partial \psi}{\partial I_2} \frac{\partial I_2}{\partial E_\alpha}
            \frac{\partial E_\alpha}{\partial \lambda_\alpha}
            + \frac{\partial \psi}{\partial I_3} \frac{\partial I_3}{\partial E_\alpha}
            \frac{\partial E_\alpha}{\partial \lambda_\alpha}


    Furthermore, the second partial derivatives of the strain energy density w.r.t. the
    principal stretches, necessary for the principal components of the elastic tangent
    moduli, are carried out. This is done in two steps: first, the second partial
    derivatives w.r.t. the principal strain components are carried out, followed by the
    projection to the derivatives w.r.t. the principal stretches.

    ..  math::

        \frac{\partial^2 \psi}{\partial E_\alpha~\partial E_\beta} &=
            \frac{\partial^2 \psi}{\partial I_1~\partial I_1}
            \frac{\partial I_1}{\partial E_\alpha}
            \frac{\partial I_1}{\partial E_\beta}
            +
            \frac{\partial^2 \psi}{\partial I_2~\partial I_2}
            \frac{\partial I_2}{\partial E_\alpha}
            \frac{\partial I_2}{\partial E_\beta}
            +
            \frac{\partial^2 \psi}{\partial I_3~\partial I_3}
            \frac{\partial I_3}{\partial E_\alpha}
            \frac{\partial I_3}{\partial E_\beta}

            &+
            \frac{\partial^2 \psi}{\partial I_1~\partial I_2}
            \left(
                \frac{\partial I_1}{\partial E_\alpha}
                \frac{\partial I_2}{\partial E_\beta}
                +
                \frac{\partial I_2}{\partial E_\alpha}
                \frac{\partial I_1}{\partial E_\beta}
            \right)

            &+
            \frac{\partial^2 \psi}{\partial I_2~\partial I_3}
            \left(
                \frac{\partial I_2}{\partial E_\alpha}
                \frac{\partial I_3}{\partial E_\beta}
                +
                \frac{\partial I_3}{\partial E_\alpha}
                \frac{\partial I_2}{\partial E_\beta}
            \right)

            &+
            \frac{\partial^2 \psi}{\partial I_1~\partial I_3}
            \left(
                \frac{\partial I_1}{\partial E_\alpha}
                \frac{\partial I_3}{\partial E_\beta}
                +
                \frac{\partial I_3}{\partial E_\alpha}
                \frac{\partial I_1}{\partial E_\beta}
            \right)

            &+
            \frac{\partial \psi}{\partial I_1}
            \frac{\partial^2 I_1}{\partial E_\alpha~\partial E_\beta}
            +
            \frac{\partial \psi}{\partial I_2}
            \frac{\partial^2 I_1}{\partial E_\alpha~\partial E_\beta}
            +
            \frac{\partial \psi}{\partial I_3}
            \frac{\partial^2 I_1}{\partial E_\alpha~\partial E_\beta}

    ..  math::

        \frac{\partial^2 \psi}{\partial \lambda_\alpha~\partial \lambda_\beta} =
            \frac{\partial E_\alpha}{\partial \lambda_\alpha}
            \frac{\partial^2 \psi}{\partial E_\alpha~\partial E_\beta}
            \frac{\partial E_\beta} {\partial \lambda_\beta}
            +
            \left(
            \frac{\partial \psi}{\partial I_1} \frac{\partial I_1}{\partial E_\alpha}
            +
            \frac{\partial \psi}{\partial I_2} \frac{\partial I_2}{\partial E_\alpha}
            +
            \frac{\partial \psi}{\partial I_3} \frac{\partial I_3}{\partial E_\alpha}
            \right)
            \frac{\partial^2 E_\alpha}{\partial \lambda_\alpha \partial \lambda_\alpha}

    """

    def __init__(self, material, fun, **kwargs):
        """Initialize the generalized invariant-based isotropic hyperelastic material
        formulation."""

        self.material = material
        self.strain = fun
        self.kwargs = kwargs

        E, dEdλ, d2Edλdλ, E0, dEdλ0, d2Edλdλ0 = self.strain(1, **self.kwargs)

        # normalize invariants
        c1_upper = 3 / 2 * (d2Edλdλ0 + dEdλ0)
        c1_lower = 3 / 2 * (d2Edλdλ + dEdλ)

        c2_upper = 3 * (d2Edλdλ0 + dEdλ0) * E0 - 3 / 2 * dEdλ0**2
        c2_lower = 3 * (d2Edλdλ + dEdλ) * E - 3 / 2 * dEdλ**2

        self.c1 = c1_upper / c1_lower
        self.c2 = c2_upper / c2_lower

        self.I10 = 3 * E
        self.I20 = 3 * E**2
        self.I30 = E**3

    def gradient(self, stretches, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the principal stretches."""

        # principal strains
        self.E, self.dEdλ, self.d2Edλdλ = self.strain(stretches, **self.kwargs)[:3]
        E = self.E

        # strain invariants
        I1 = E[0] + E[1] + E[2]
        I2 = E[0] * E[1] + E[1] * E[2] + E[2] * E[0]
        I3 = E[0] * E[1] * E[2]

        self.I1 = self.c1 * I1 - self.I10 * (self.c1 - 1)
        self.I2 = self.c2 * I2 - self.I20 * (self.c2 - 1)
        self.I3 = I3

        self.dWdI1, self.dWdI2, self.dWdI3, statevars_new = self.material.gradient(
            self.I1, self.I2, self.I3, statevars
        )

        Eβ = E[[1, 0, 0]]
        Eγ = E[[2, 2, 1]]

        self.dI1dE = self.c1 * np.ones_like(E)
        self.dI2dE = self.c2 * (Eβ + Eγ)
        self.dI3dE = Eβ * Eγ

        dWdλ = np.zeros_like(stretches)

        if self.dWdI1 is not None:
            dWdλ += self.dWdI1 * self.dI1dE * self.dEdλ

        if self.dWdI2 is not None:
            dWdλ += self.dWdI2 * self.dI2dE * self.dEdλ

        if self.dWdI3 is not None:
            dWdλ += self.dWdI3 * self.dI3dE * self.dEdλ

        return dWdλ, statevars_new

    def hessian(self, stretches, statevars):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the principal stretches."""

        dWdλα, statevars_new = self.gradient(stretches, statevars)
        (
            d2WdI1dI1,
            d2WdI2dI2,
            d2WdI3dI3,
            d2WdI1dI2,
            d2WdI2dI3,
            d2WdI1dI3,
        ) = self.material.hessian(self.I1, self.I2, self.I3, statevars)

        dI1dE = self.dI1dE
        dI2dE = self.dI2dE
        dI3dE = self.dI3dE
        dEdλ = self.dEdλ
        d2Edλdλ = self.d2Edλdλ

        Eγ = [2, 0, 1]

        d2WdEαdEβ = np.zeros((6, *dWdλα.shape[1:]))

        α = [0, 1, 2, 0, 1, 0]
        β = [0, 1, 2, 1, 2, 2]

        if d2WdI1dI1 is not None:
            d2WdEαdEβ += d2WdI1dI1 * dI1dE[α] * dI1dE[β]

        if d2WdI2dI2 is not None:
            d2WdEαdEβ += d2WdI2dI2 * dI2dE[α] * dI2dE[β]

        if d2WdI3dI3 is not None:
            d2WdEαdEβ += d2WdI3dI3 * dI3dE[α] * dI3dE[β]

        if d2WdI1dI2 is not None:
            d2WdEαdEβ += d2WdI1dI2 * (dI1dE[α] * dI2dE[β] + dI2dE[α] * dI1dE[β])

        if d2WdI2dI3 is not None:
            d2WdEαdEβ += d2WdI2dI3 * (dI2dE[α] * dI3dE[β] + dI3dE[α] * dI2dE[β])

        if d2WdI1dI3 is not None:
            d2WdEαdEβ += d2WdI1dI3 * (dI1dE[α] * dI3dE[β] + dI3dE[α] * dI1dE[β])

        # if self.dWdI1 is not None:
        #     d2I1dEαdEβ = 0
        #     d2WdEαdEβ[3:] += self.dWdI2 * d2I1dEαdEβ

        if self.dWdI2 is not None:
            d2I2dEαdEβ = self.c2
            d2WdEαdEβ[3:] += self.dWdI2 * d2I2dEαdEβ

        if self.dWdI3 is not None:
            d2I3dEαdEβ = Eγ
            d2WdEαdEβ[3:] += self.dWdI3 * d2I3dEαdEβ

        # in-place modification (avoid the creation of a new array)
        d2Wdλαdλβ = d2WdEαdEβ
        d2Wdλαdλβ *= dEdλ[α] * dEdλ[β]

        if self.dWdI1 is not None:
            d2Wdλαdλβ[:3] += self.dWdI1 * dI1dE * d2Edλdλ

        if self.dWdI2 is not None:
            d2Wdλαdλβ[:3] += self.dWdI2 * dI2dE * d2Edλdλ

        if self.dWdI3 is not None:
            d2Wdλαdλβ[:3] += self.dWdI3 * dI3dE * d2Edλdλ

        return d2Wdλαdλβ
