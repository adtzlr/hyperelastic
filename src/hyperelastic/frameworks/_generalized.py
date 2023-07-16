from ..models.stretches import GeneralizedInvariantsModel
from ._stretches import Stretches


class GeneralizedInvariants(Stretches):
    r"""Generalized-invariants isotropic hyperelastic material formulation based on the
    principal stretches. The generalized invariants

    ..  math::

        I_1 &= E_1 + E_2 + E_3

        I_2 &= E_1 E_2 + E_2 E_3 + E_1 E_3

        I_3 &= E_1 E_2 E_3

    are related to a one-dimensional strain-stretch relation.

    ..  math::

        E_\alpha &= f(\lambda_\alpha)

        E'_\alpha &= f'(\lambda_\alpha) = \frac{\partial f(\lambda_\alpha)}
            {\partial \lambda_\alpha}

        E''_\alpha &= f''(\lambda_\alpha) = \frac{\partial^2 f(\lambda_\alpha)}
            {\partial \lambda_\alpha~\partial \lambda_\alpha}

    The first partial derivatives of the strain energy function w.r.t. the invariants

    ..  math::

        \psi_{,1} &= \frac{\partial \psi}{\partial I_1}

        \psi_{,2} &= \frac{\partial \psi}{\partial I_2}

        \psi_{,3} &= \frac{\partial \psi}{\partial I_3}

    and the partial derivatives of the invariants w.r.t. the principal stretches are
    defined.

    ..  math::

        \frac{\partial I_1}{\partial E_\alpha} &= 1

        \frac{\partial I_2}{\partial E_\alpha} &= E_\beta + E_\gamma

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
