from ..models.stretches import GeneralizedInvariantsModel
from ._stretches import Stretches


class GeneralizedInvariants(Stretches):
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

    To be consistent with linear elasticity, the invariants are scaled by deformation-
    independent coefficients of normalization.

    ..  math::

        I_1 &= c_1 J_1(E_\alpha(\lambda_\alpha))
             - J_1(E_\alpha(\lambda_\alpha=1)) (c_1 - 1)

        I_2 &= c_2 J_2(E_\alpha(\lambda_\alpha))
             - J_2(E_\alpha(\lambda_\alpha=1)) (c_2 - 1)

        I_3 &= J_3

    Note that they are only applied to the first and second invariant, as the third
    invariant does not contribute to the strain energy function at the undeformed state.
    The second partial derivative of the strain w.r.t. the stretch must be provided for
    a reference strain (at the undeformed state).

    ..  math::

        c_1 &= \frac{E''_{ref}(\lambda=1) / 3}{
            \left( E''(\lambda=1) + E'(\lambda=1) \right) / 2}

        c_2 &= \frac{E''_{ref}(\lambda=1, k=2) / 3}{
            \left( E''(\lambda=1) + E'(\lambda=1) \right) E(\lambda=1)
            - E'^2(\lambda=1) / 2}

    The first partial derivatives of the strain energy function w.r.t. the invariants

    ..  math::

        \psi_{,1} &= \frac{\partial \psi}{\partial I_1}

        \psi_{,2} &= \frac{\partial \psi}{\partial I_2}

        \psi_{,3} &= \frac{\partial \psi}{\partial I_3}

    and the partial derivatives of the invariants w.r.t. the principal stretches are
    defined.

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

    def __init__(self, material, fun, nstatevars=0, parallel=False, **kwargs):
        """Initialize the generalized invariant-based isotropic hyperelastic material
        formulation."""

        model = GeneralizedInvariantsModel(material, fun, **kwargs)
        super().__init__(material=model, nstatevars=nstatevars, parallel=parallel)
