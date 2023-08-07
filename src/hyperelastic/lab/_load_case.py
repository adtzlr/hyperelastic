import numpy as np


class IncompressibleHomogeneousStretch:
    r"""An incompressible homogeneous stretch load case with a longitudinal stretch and
    perpendicular transverse stretches in principal directions. This class is intended
    to be subclassed by another class with a `.defgrad()` method for the evaluation of
    the deformation gradient as utilized by the :class:`Uniaxial <.lab.Uniaxial>`,
    :class:`Planar <.lab.Planar>` and :class:`Biaxial <.lab.Biaxial>` load cases.

    Notes
    -----
    The Cauchy stress for an incompressible material is given by

    ..  math::
        \boldsymbol{\sigma} = \boldsymbol{\sigma}' + p \boldsymbol{I}

    where the Cauchy stress is converted to the first Piola-Kirchhoff stress tensor.

    ..  math::
        \boldsymbol{P} = J \boldsymbol{\sigma} \boldsymbol{F}^{-T}

    The deformation gradient and its determinant are evaluated for the
    homogeneous incompressible deformation.

    ..  math::
        \boldsymbol{F} &= \text{diag} \left(\begin{bmatrix}
            \lambda_1 & \lambda_2 & \lambda_3 \end{bmatrix} \right)

        J &= \lambda_1 \lambda_2 \lambda_3 = 1

    This enables the evaluation of the normal force per undeformed area, where
    quantities in the traction-free transverse direction are denoted with a subscript
    :math:`(\bullet)_t`.

    ..  math::
        \frac{N}{A} = P - P_t \frac{\lambda_t}{\lambda}

    """

    def stress(self, F, P, axis=0, traction_free=-1):
        r"""Normal force per undeformed area for a given deformation gradient of an
        incompressible deformation and the first Piola-Kirchhoff stress tensor.

        Parameters
        ----------
        F : ndarray
            The deformation gradient.
        P : ndarray
            The first Piola-Kirchhoff stress tensor.
        axis : int, optional
            The primary axis where the longitudinal stretch is applied on (default is
            0).
        traction_free : int, optional
            The secondary axis where the traction-free transverse stretch results from
            the constraint of incompressibility (default is -1).

        Returns
        -------
        ndarray
            The one-dimensional normal force per undeformed area.

        """

        i = axis
        j = traction_free

        return P[i, i] - P[j, j] * F[j, j] / F[i, i]


class Uniaxial(IncompressibleHomogeneousStretch):
    r"""Incompressible uniaxial tension/compression load case.

    ..  math::
        \boldsymbol{F} = \text{diag} \left(\begin{bmatrix}
            \lambda & \frac{1}{\sqrt{\lambda}} & \frac{1}{\sqrt{\lambda}} \end{bmatrix}
            \right)

    """

    def __init__(self, label=None):
        if label is None:
            label = "Uniaxial Tension"
        self.label = label

    def defgrad(self, stretch):
        "Return the Deformation Gradient tensor from given stretches."

        x = stretch
        y = 1 / np.sqrt(stretch)
        z = np.zeros_like(stretch)

        return np.array([[x, z, z], [z, y, z], [z, z, y]])


class Biaxial(IncompressibleHomogeneousStretch):
    r"""Incompressible biaxial tension/compression load case.

    ..  math::
        \boldsymbol{F} = \text{diag} \left(\begin{bmatrix}
            \lambda & \lambda & \frac{1}{\lambda^2} \end{bmatrix}
            \right)

    """

    def __init__(self, label=None):
        if label is None:
            label = "Biaxial Tension"
        self.label = label

    def defgrad(self, stretch):
        "Return the Deformation Gradient tensor from given stretches."

        x = stretch
        y = 1 / stretch**2
        z = np.zeros_like(stretch)

        return np.array([[x, z, z], [z, x, z], [z, z, y]])


class Planar(IncompressibleHomogeneousStretch):
    r"""Incompressible planar (shear) tension/compression load case.

    ..  math::
        \boldsymbol{F} = \text{diag} \left(\begin{bmatrix}
            \lambda & 1 & \frac{1}{\lambda} \end{bmatrix}
            \right)

    """

    def __init__(self, label=None):
        if label is None:
            label = "Planar Tension"
        self.label = label

    def defgrad(self, stretch):
        "Return the Deformation Gradient tensor from given stretches."

        x = stretch
        y = 1 / stretch
        o = np.ones_like(stretch)
        z = np.zeros_like(stretch)

        return np.array([[x, z, z], [z, o, z], [z, z, y]])
