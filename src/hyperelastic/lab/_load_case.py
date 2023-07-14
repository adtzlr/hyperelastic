import numpy as np


class Uniaxial:
    "Incompressible uniaxial tensions/compression load case."

    def defgrad(self, stretch):
        "Return the Deformation Gradient tensor from given stretches."
        x = stretch
        y = 1 / np.sqrt(stretch)
        z = np.zeros_like(stretch)
        return np.array([[x, z, z], [z, y, z], [z, z, y]])

    def stress(self, F, P):
        """Normal force per undeformed area for given deformation gradient and first
        Piola-Kirchhoff stress tensor."""
        return P[0, 0] - P[2, 2] * F[2, 2] / F[0, 0]


class Biaxial:
    "Incompressible Biaxial tensions/compression load case."

    def defgrad(self, stretch):
        "Return the Deformation Gradient tensor from given stretches."
        x = stretch
        y = 1 / stretch**2
        z = np.zeros_like(stretch)
        return np.array([[x, z, z], [z, x, z], [z, z, y]])

    def stress(self, F, P):
        """Normal force per undeformed area for given deformation gradient and first
        Piola-Kirchhoff stress tensor."""
        return P[0, 0] - P[2, 2] * F[2, 2] / F[0, 0]
