import numpy as np


def asvoigt(A, mode=2):
    r"""Convert a three-dimensional tensor from full array-storage into reduced
    symmetric (Voigt-notation) vector/matrix storage.

    Parameters
    ----------
    A : np.ndarray
        A three-dimensional second- or fourth-order tensor in full array-storage.
    mode : int, optional
        The mode, 2 for second-order and 4 for fourth-order tensors (default is 2).

    Returns
    -------
    np.ndarray
        A three-dimensional second- or fourth-order tensor in reduced symmetric (Voigt)
        vector/matrix storage.

    Notes
    -----
    This is the inverse operation of :func:`astensor`.

    For a symmetric 3x3 second-order tensor :math:`C_{ij} = C_{ji}`, the upper triangle
    entries are inserted into a 6x1 vector, starting from the main diagonal, followed by
    the consecutive next upper diagonals.

    ..  math::

        \boldsymbol{C} = \begin{bmatrix}
            C_{11} & C_{12} & C_{13} \\
            C_{12} & C_{22} & C_{23} \\
            C_{13} & C_{23} & C_{33}
        \end{bmatrix} \qquad \longrightarrow \boldsymbol{C} = \begin{bmatrix}
            C_{11} & C_{22} & C_{33} & C_{12} & C_{23} & C_{13}
        \end{bmatrix}^T

    For a (at least minor) symmetric 3x3x3x3 fourth-order tensor :math:`A_{ijkl} =
    A_{jikl} = A_{ijlk} = A_{jilk}`, rearranged to 9x9, the upper triangle entries are
    inserted into a 6x6 matrix, starting from the main diagonal, followed by the
    consecutive next upper diagonals.

    ..  math::

        \begin{bmatrix}
            A_{1111} & A_{1112} & A_{1113} &
            A_{1121} & A_{1122} & A_{1123} &
            A_{1131} & A_{1132} & A_{1133} \\
            %
            A_{1211} & A_{1212} & A_{1213} &
            A_{1221} & A_{1222} & A_{1223} &
            A_{1231} & A_{1232} & A_{1233} \\
            %
            \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots \\
            A_{3111} & A_{3112} & A_{3113} &
            A_{3121} & A_{3122} & A_{3123} &
            A_{3131} & A_{3132} & A_{3133}
            %
        \end{bmatrix} \qquad

        \longrightarrow \mathbb{A} = \begin{bmatrix}
            A_{1111} & A_{1122} & A_{1133} & A_{1112} & A_{1123} & A_{1113} \\
            A_{2211} & A_{2222} & A_{2233} & A_{2212} & A_{2223} & A_{2213} \\
             \dots   &  \dots   &  \dots   &  \dots   &  \dots   &  \dots   \\
            A_{1311} & A_{1322} & A_{1333} & A_{1312} & A_{1323} & A_{1313}
        \end{bmatrix}

    Examples
    --------
    >>> import hyperelastic.math as hm
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> asvoigt(C, mode=2)
    array([1. , 1.1, 1.2, 1.3, 1.4, 1.5])

    >>> A = np.einsum("ij,kl", C, C)
    >>> asvoigt(A, mode=4)
    array([[1.  , 1.1 , 1.2 , 1.3 , 1.4 , 1.5 ],
           [1.1 , 1.21, 1.32, 1.43, 1.54, 1.65],
           [1.2 , 1.32, 1.44, 1.56, 1.68, 1.8 ],
           [1.3 , 1.43, 1.56, 1.69, 1.82, 1.95],
           [1.4 , 1.54, 1.68, 1.82, 1.96, 2.1 ],
           [1.5 , 1.65, 1.8 , 1.95, 2.1 , 2.25]])

    """

    if mode == 2:  # 3x3 symmetric tensor
        i = [0, 1, 2, 0, 1, 0]
        j = [0, 1, 2, 1, 2, 2]

        return A[i, j]

    elif mode == 4:  # 3x3x3x3 major-symmetric tensor
        B = np.zeros((6, 6, *A.shape[4:]))
        for i in range(3):
            for j in range(3):
                B[i, j] = A[i, i, j, j]

        B[3, 3] = A[0, 1, 0, 1]
        B[4, 4] = A[1, 2, 1, 2]
        B[5, 5] = A[0, 2, 0, 2]

        for i in range(3):
            B[i, 3] = A[i, i, 0, 1]
            B[i, 4] = A[i, i, 1, 2]
            B[i, 5] = A[i, i, 0, 2]
            B[3, i] = A[0, 1, i, i]
            B[4, i] = A[1, 2, i, i]
            B[5, i] = A[0, 2, i, i]

        B[3, 4] = A[0, 1, 1, 2]
        B[3, 5] = A[0, 1, 0, 2]

        B[4, 3] = A[1, 2, 0, 1]
        B[4, 5] = A[1, 2, 0, 2]

        B[5, 3] = A[0, 2, 0, 1]
        B[5, 4] = A[0, 2, 1, 2]

        return B


def astensor(A, mode=2):
    r"""Convert a three-dimensional tensor from symmetric (Voigt-notation) vector/matrix
    storage into full array-storage.

    Parameters
    ----------
    A : np.ndarray
        A three-dimensional second- or fourth-order tensor in reduced symmetric (Voigt)
        vector/matrix storage.
    mode : int, optional
        The mode, 2 for second-order and 4 for fourth-order tensors (default is 2).

    Returns
    -------
    np.ndarray
        A three-dimensional second- or fourth-order tensor in full array-storage.

    Notes
    -----
    This is the inverse operation of :func:`asvoigt`.

    For a symmetric 3x3 second-order tensor :math:`C_{ij} = C_{ji}`, the entries are
    re-created from a 6x1 vector.

    ..  math::

        \boldsymbol{C} = \begin{bmatrix}
            C_{11} & C_{22} & C_{33} & C_{12} & C_{23} & C_{13}
        \end{bmatrix}^T \longrightarrow
        %
        \boldsymbol{C} = \begin{bmatrix}
            C_{11} & C_{12} & C_{13} \\
            C_{12} & C_{22} & C_{23} \\
            C_{13} & C_{23} & C_{33}
        \end{bmatrix} \qquad

    For a (at least minor) symmetric 3x3x3x3 fourth-order tensor :math:`A_{ijkl} =
    A_{jikl} = A_{ijlk} = A_{jilk}`, the entries are re-created from
    a 6x6 matrix.

    ..  math::

        \mathbb{A} = \begin{bmatrix}
            A_{1111} & A_{1122} & A_{1133} & A_{1112} & A_{1123} & A_{1113} \\
            A_{2211} & A_{2222} & A_{2233} & A_{2212} & A_{2223} & A_{2213} \\
             \dots   &  \dots   &  \dots   &  \dots   &  \dots   &  \dots   \\
            A_{1311} & A_{1322} & A_{1333} & A_{1312} & A_{1323} & A_{1313}
        \end{bmatrix}

        \longrightarrow \begin{bmatrix}
            A_{1111} & A_{1112} & A_{1113} &
            A_{1121} & A_{1122} & A_{1123} &
            A_{1131} & A_{1132} & A_{1133} \\
            %
            A_{1211} & A_{1212} & A_{1213} &
            A_{1221} & A_{1222} & A_{1223} &
            A_{1231} & A_{1232} & A_{1233} \\
            %
            \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots \\
            A_{3111} & A_{3112} & A_{3113} &
            A_{3121} & A_{3122} & A_{3123} &
            A_{3131} & A_{3132} & A_{3133}
            %
        \end{bmatrix} \qquad

    Examples
    --------
    >>> import hyperelastic.math as hm
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> C6 = asvoigt(C, mode=2)
    >>> D = astensor(C6, mode=2)
    >>> np.allclose(C, D)
    True

    >>> D
    array([[1. , 1.3, 1.5],
           [1.3, 1.1, 1.4],
           [1.5, 1.4, 1.2]])

    >>> A = np.einsum("ij,kl", C, C)
    >>> A66 = asvoigt(A, mode=4)
    >>> B = astensor(A66, mode=4)
    >>> np.allclose(A, B)
    True

    """

    if mode == 2:  # second order tensor of shape 6 to 3x3
        a = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)
        return A[a]

    elif mode == 4:  # fourth order tensor of shape 6x6 to 3x3x3x3
        a = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2])
        i, j = np.meshgrid(a, a)
        return A[i.reshape(3, 3, 3, 3), j.reshape(3, 3, 3, 3)]


def tril_from_triu(A, dim=6, inplace=True):
    "Copy upper triangle values to lower triangle values of a nxn tensor inplace."
    B = A
    if not inplace:
        B = A.copy()

    i, j = np.triu_indices(dim, 1)
    B[j, i] = A[i, j]
    return B


def triu_from_tril(A, dim=6, inplace=True):
    "Copy lower triangle values to upper triangle values of a nxn tensor inplace."
    B = A
    if not inplace:
        B = A.copy()

    i, j = np.tril_indices(dim, -1)
    B[j, i] = A[i, j]
    return B


def trace(A):
    "Return the sum of the diagonal values of a 3x3 tensor."
    return np.sum(A[:3], axis=0)


def transpose(A):
    "Return the major transpose of a fourth order tensor in Voigt-storage."
    return np.einsum("ij...->ji...", A)


def det(A):
    "The determinant of a symmetric 3x3 tensor in Voigt-storage (rule of Sarrus)."
    return (
        A[0] * A[1] * A[2]
        + 2 * A[3] * A[4] * A[5]
        - A[4] ** 2 * A[0]
        - A[5] ** 2 * A[1]
        - A[3] ** 2 * A[2]
    )


def inv(A, determinant=None):
    """The inverse of a symmetric 3x3 tensor in Voigt-storage with optional provided
    determinant."""

    detAinvA = np.zeros_like(A)

    if determinant is None:
        detA = det(A)
    else:
        detA = determinant

    detAinvA[0] = A[1] * A[2] - A[4] ** 2
    detAinvA[1] = A[0] * A[2] - A[5] ** 2
    detAinvA[2] = A[0] * A[1] - A[3] ** 2

    detAinvA[3] = A[4] * A[5] - A[3] * A[2]
    detAinvA[4] = A[3] * A[5] - A[0] * A[4]
    detAinvA[5] = A[3] * A[4] - A[1] * A[5]

    return detAinvA / detA


def dot(A, B, mode=(2, 2)):
    "The dot-product of two symmetric 3x3 tensors in Voigt-storage."

    if mode == (2, 2):
        trax = np.broadcast_shapes(A.shape[1:], B.shape[1:])
        C = np.zeros((3, 3, *trax))
        C[0, 0] = A[0] * B[0] + A[3] * B[3] + A[5] * B[5]
        C[1, 1] = A[3] * B[3] + A[1] * B[1] + A[4] * B[4]
        C[2, 2] = A[4] * B[4] + A[5] * B[5] + A[2] * B[2]
        C[0, 1] = A[0] * B[3] + A[3] * B[1] + A[5] * B[4]
        C[1, 2] = A[3] * B[5] + A[1] * B[4] + A[4] * B[2]
        C[0, 2] = A[0] * B[5] + A[3] * B[4] + A[5] * B[2]
        C[1, 0] = A[3] * B[0] + A[1] * B[3] + A[4] * B[5]
        C[2, 1] = A[5] * B[3] + A[4] * B[1] + A[2] * B[4]
        C[2, 0] = A[5] * B[0] + A[4] * B[3] + A[2] * B[5]

    return C


def eye(A):
    "A 3x3 tensor in Voigt-storage with ones on the diagonal and zeros elsewhere."
    trax = np.ones(len(A.shape[1:]), dtype=int)
    return np.array([1, 1, 1, 0, 0, 0], dtype=float).reshape(6, *trax)


def ddot(A, B, mode=(2, 2)):
    "The double-contraction of two symmetric 3x3 tensors in Voigt-storage."
    weights = np.array([1, 1, 1, 2, 2, 2])
    if mode == (2, 2):
        return np.einsum("i...,i...,i->...", A, B, weights)
    elif mode == (4, 4):
        return np.einsum("ik...,kj...,k->ij...", A, B, weights)
    elif mode == (4, 2):
        return np.einsum("ij...,j...,j->i...", A, B, weights)
    elif mode == (2, 4):
        return np.einsum("i...,ij...,i->j...", A, B, weights)


def dya(A, B):
    "The dyadic (outer) product."
    return A.reshape(-1, 1, *A.shape[1:]) * B.reshape(1, -1, *B.shape[1:])


def dev(A):
    "The deviatoric part of a 3x3 tensor."
    return A - trace(A) / 3 * eye(A)


def cdya_ik(A, B):
    r"""The inner-crossed dyadic product of two symmetric second-order
    tensors in reduced vector storage, where the inner indices (the right index of the
    first tensor and the left index of the second tensor) are interchanged.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.

    Returns
    -------
    np.ndarray
        Inner-crossed dyadic product in full-array storage.

    Notes
    -----
    The result of the inner-crossed dyadic product of two symmetric second order tensors
    is a major- (but not minor-) symmetric fourth-order tensor. This is also the case
    for :math:`\boldsymbol{A} = \boldsymbol{B}`.

    ..  math::

        \mathbb{C} &= \boldsymbol{A} \overline{\otimes} \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{ik}~B_{jl}

    Examples
    --------
    >>> import hyperelastic.math as hm
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> A = np.einsum("ik,jl", C, D)

    >>> C6 = asvoigt(C, mode=2)
    >>> D6 = asvoigt(D, mode=2)

    >>> np.allclose(A, cdya_ik(C6, D6))
    True

    """
    i, j = [a.ravel() for a in np.indices((9, 9))]

    a = np.array(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    )
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    ik = b[i, k].reshape(3, 3, 3, 3)
    jl = b[j, l].reshape(3, 3, 3, 3)

    return A[ik] * B[jl]


def cdya_il(A, B):
    r"""The right-crossed dyadic product of two symmetric second-order
    tensors in reduced vector storage, where the right indices of the two tensors are
    interchanged.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.

    Returns
    -------
    np.ndarray
        Right-crossed dyadic product in full-array storage.

    Notes
    -----
    The result of the right-crossed dyadic product of two symmetric second
    order tensors is a non-symmetric fourth-order tensor. In case of
    :math:`\boldsymbol{A} = \boldsymbol{B}`, the fourth-order tensor is major-symmetric.

    ..  math::

        \mathbb{C} &= \boldsymbol{A} {\otimes\small{|}} \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{il}~B_{kj}

    Examples
    --------
    >>> import hyperelastic.math as hm
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> A = np.einsum("il,kj", C, D)

    >>> C6 = asvoigt(C, mode=2)
    >>> D6 = asvoigt(D, mode=2)

    >>> np.allclose(A, cdya_il(C6, D6))
    True

    """
    i, j = [a.ravel() for a in np.indices((9, 9))]

    a = np.array(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    )
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    il = b[i, l].reshape(3, 3, 3, 3)
    kj = b[k, j].reshape(3, 3, 3, 3)

    return A[il] * B[kj]


def cdya(A, B):
    r"""The full-symmetric crossed-dyadic product of two symmetric second-order tensors
    in reduced vector storage.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.

    Returns
    -------
    np.ndarray
        Symmetric crossed-dyadic product in reduced matrix storage.

    Notes
    -----
    The result of the symmetric crossed-dyadic product of two symmetric second order
    tensors is a minor- and major-symmetric fourth-order tensor.

    ..  math::

        \mathbb{C} &= \frac{1}{2} \left(
            \boldsymbol{A} \odot \boldsymbol{B} + \boldsymbol{B} \odot \boldsymbol{S}
        \right)

        \mathbb{C} &= \frac{1}{4} \left(
            \boldsymbol{A} \overline{\otimes} \boldsymbol{B} +
            \boldsymbol{A} \underline{\otimes} \boldsymbol{B} +
            \boldsymbol{B} \overline{\otimes} \boldsymbol{A} +
            \boldsymbol{B} \underline{\otimes} \boldsymbol{A}
        \right)

        \mathbb{C}_{ijkl} &= \frac{1}{4} \left(
            A_{ik}~B_{jl} + A_{il}~B_{kj} + B_{ik}~A_{jl} + B_{il}~A_{kj}
        \right)

    Examples
    --------
    >>> import hyperelastic.math as hm
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> CD = (np.einsum("ik,jl", C, D) + np.einsum("il,kj", C, D)) / 2
    >>> DC = (np.einsum("ik,jl", D, C) + np.einsum("il,kj", D, C)) / 2

    >>> A = (CD + DC) / 2

    >>> C6 = asvoigt(C, mode=2)
    >>> D6 = asvoigt(D, mode=2)

    >>> np.allclose(A, astensor(cdya(C6, D6), mode=4))
    True

    >>> np.allclose(A, astensor(cdya(D6, C6), mode=4))
    True

    """

    i, j = [a.ravel() for a in np.indices((6, 6))]

    a = np.array([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)])
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    ik = b[i, k].reshape(6, 6)
    jl = b[j, l].reshape(6, 6)

    il = b[i, l].reshape(6, 6)
    kj = b[k, j].reshape(6, 6)

    C = (A[ik] * B[jl] + A[il] * B[kj]) / 2

    if A is not B:
        C += (B[ik] * A[jl] + B[il] * A[kj]) / 2
        C /= 2

    return C


def eigh(A, fun=None):
    "Eigenvalues and -bases of matrix A."
    wA, vA = np.linalg.eigh(astensor(A).T)
    MA = np.einsum("...ia,...ja->...ija", vA, vA)
    if fun is not None:
        wA = fun(wA)
    return wA.T, np.array([asvoigt(M) for M in MA.T])
