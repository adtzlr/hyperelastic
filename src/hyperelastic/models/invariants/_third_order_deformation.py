class ThirdOrderDeformation:
    r"""Third Order Deformation isotropic hyperelastic material formulation based on the
    first and second invariant of the right Cauchy-Green deformation tensor. The strain
    energy density per unit undeformed volume is given as a sum of multivariate
    polynomials.

    ..  math::

        \psi(I_1, I_2) =
            \sum_{(i+j) \ge 1}^{(i+2j) \le 3} C_{ij}~(I_1 - 3)^i (I_2 - 3)^j

    The first partial derivatives of the strain energy density w.r.t. the
    invariants are given below.

    ..  math::

        \frac{\partial \psi}{\partial I_1} &=
            \sum_{i \ge 1} C_{ij}~i~(I_1 - 3)^{i-1} (I_2 - 3)^j

        \frac{\partial \psi}{\partial I_2} &=
            \sum_{j \ge 1} C_{ij}~(I_1 - 3)^i~j~(I_2 - 3)^{j-1}

    Furthermore, the second partial derivatives of the strain energy density w.r.t. the
    invariants are carried out.

    ..  math::

        \frac{\partial^2 \psi}{\partial I_1~\partial I_1} &=
            \sum_{i \ge 2} C_{ij}~i~(i-1)~(I_1 - 3)^{i-2} (I_2 - 3)^j

        \frac{\partial^2 \psi}{\partial I_2~\partial I_2} &=
            \sum_{j \ge 2} C_{ij}~(I_1 - 3)^i~j~(j-1)~(I_2 - 3)^{j-2}

        \frac{\partial^2 \psi}{\partial I_1~\partial I_2} &=
            \sum_{i \ge 1, j \ge 1} C_{ij}~i~(I_1 - 3)^{i-1}~j~(I_2 - 3)^{j-1}


    """

    def __init__(self, C10=0, C01=0, C11=0, C20=0, C30=0):
        """Initialize the Third Order Deformation material formulation with its
        parameters.
        """

        self.C10 = C10
        self.C01 = C01
        self.C11 = C11
        self.C20 = C20
        self.C30 = C30

    def gradient(self, I1, I2, I3, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the invariants."""

        dWdI1 = (
            self.C10
            + self.C11 * (I2 - 3)
            + self.C20 * 2 * (I1 - 3)
            + self.C30 * 3 * (I1 - 3) ** 2
        )
        dWdI2 = self.C01 + self.C11 * (I1 - 3)
        dWdI3 = None

        return dWdI1, dWdI2, dWdI3, statevars

    def hessian(self, I1, I2, I3, statevars):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the invariants."""

        d2WdI1I1 = self.C20 * 2 + self.C30 * 6 * (I1 - 3)
        d2WdI2I2 = None
        d2WdI3I3 = None
        d2WdI1I2 = self.C11
        d2WdI2I3 = None
        d2WdI1I3 = None

        return d2WdI1I1, d2WdI2I2, d2WdI3I3, d2WdI1I2, d2WdI2I3, d2WdI1I3
