import numpy as np

from ..math import astensor, asvoigt, cdya_ik, det, eye, inv


class Space:
    r"""A base-class for a space with a piola-transformation method to convert Total-
    Lagrange stress or elasticity tensors to other differential force- and differential
    area-vector configurations.

    Parameters
    ----------
    parallel : bool, optional
        Use a threaded version of ``numpy.einsum`` by ``einsumt`` (default is False).
    finalize : bool, optional
        Finalize the stress or elasticity tensor (default is True). If true, the stress
        or elasticity tensor are converted to full tensor storage and are transformed
        into given configurations of the differential force and differential area
        vectors. If False, no transformation is performed and no initial stress
        contribution is added to the fourth-order elasticity tensor.
    force : int or None, optional
        The configuration of the differential force vector. 0 invokes the undeformed
        configuration, None the deformed configuration (default is None).
    area : int or None, optional
        The configuration of the differential area vector. 0 invokes the undeformed
        configuration, None the deformed configuration (default is None).
    """

    def __init__(self, parallel=False, finalize=True, force=None, area=0):
        self.parallel = parallel
        self.finalize = finalize

        # configurations for the
        # * the differential force vector and
        # * differential area vector (line/area/volume)
        self.force = force
        self.area = area

        if self.parallel:
            from einsumt import einsumt

            self.einsum = einsumt
        else:
            self.einsum = np.einsum

    def piola(self, F, S, detF=None, C4=None, invC=None):
        """Convert the Total-Lagrange stress or elasticity tensor to the chosen
        configurations for the differential force and area vectors by applying a
        Piola-transformation.
        """

        if C4 is None:
            stress = S

            # piola transformation
            if self.finalize:
                stress = astensor(stress)

                # force --> deformed configuration
                if self.force is None:
                    # check if destination array stress must be broadcasted
                    new_shape = np.broadcast_shapes(F.shape, stress.shape)

                    if not np.allclose(new_shape, stress.shape):
                        # broadcast stress and make a writable contiguous copy
                        stress = np.broadcast_to(stress, new_shape).copy()

                    stress = self.einsum("iI...,IJ...->iJ...", F, stress, out=stress)

                # area --> deformed configuration
                if self.area is None:
                    if detF is None:
                        C = asvoigt(self.einsum("kI...,kJ...->IJ...", F, F))
                        I3 = det(C)
                        detF = np.sqrt(I3)

                    stress = (
                        self.einsum("jJ...,iJ...->ij...", F, stress, out=stress) / detF
                    )

            return stress

        else:
            elasticity = C4

            # piola transformation
            if self.finalize:
                elasticity = astensor(elasticity, 4)
                I3 = None

                # force --> deformed configuration
                if self.force is None:
                    # check if destination array elasticity must be broadcasted
                    new_shape = np.broadcast_shapes((3, 3, *F.shape), elasticity.shape)

                    if not np.allclose(new_shape, elasticity.shape):
                        # broadcast elasticity and make a writable contiguous copy
                        elasticity = np.broadcast_to(elasticity, new_shape).copy()

                    elasticity = self.einsum(
                        "iI...,kK...,IJKL...->iJkL...", F, F, elasticity, out=elasticity
                    )
                    # initial stress contribution
                    elasticity += cdya_ik(eye(S), S)

                # force --> undeformed configuration
                else:
                    # inverse of right Cauchy-Green deformation tensor
                    if invC is None:
                        C = asvoigt(self.einsum("kI...,kJ...->IJ...", F, F))
                        I3 = det(C)
                        invC = inv(C, determinant=I3)

                    # initial stress contribution
                    elasticity += cdya_ik(invC, S)

                # area --> deformed configuration
                if self.area is None:
                    # determinant of deformation gradient
                    if detF is None:
                        # determinant of right Cauchy-Green deformation tensor
                        if I3 is None:
                            C = asvoigt(self.einsum("kI...,kJ...->IJ...", F, F))
                            I3 = det(C)

                        detF = np.sqrt(I3)

                    elasticity = (
                        self.einsum(
                            "jJ...,lL...,iJkL...->ijkl...",
                            F,
                            F,
                            elasticity,
                            out=elasticity,
                        )
                        / detF
                    )

            return elasticity
