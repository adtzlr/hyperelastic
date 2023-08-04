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
    >>> from hyperelastic.math import asvoigt
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
    >>> from hyperelastic.math import asvoigt, astensor
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
        i, j = np.meshgrid(a, a, indexing="ij")
        return A[i.reshape(3, 3, 3, 3), j.reshape(3, 3, 3, 3)]


def tril_from_triu(A, dim=6, out=None):
    "Copy upper triangle values to lower triangle values of a nxn tensor inplace."

    if out is None:
        out = A.copy()

    i, j = np.triu_indices(dim, 1)
    out[j, i] = A[i, j]
    return out


def triu_from_tril(A, dim=6, out=None):
    "Copy lower triangle values to upper triangle values of a nxn tensor inplace."

    if out is None:
        out = A.copy()

    i, j = np.tril_indices(dim, -1)
    out[j, i] = A[i, j]
    return out


def trace(A):
    r"""The trace of a symmetric second-order tensor in reduced vector storage as the
    sum of the main diagonal.

    Parameters
    ----------
    A : np.ndarray
        Symmetric second-order tensor in reduced vector storage.

    Returns
    -------
    np.ndarray
        Trace of a symmetric second-order tensor in reduced vector storage.

    Notes
    -----

    ..  math::

        \text{tr}\left(\boldsymbol{C}\right) &= \boldsymbol{C} : \boldsymbol{I}
            = C_{11} + C_{22} + C_{33}

        C_{kk} &= C_{ij} : \delta_{ij}

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, trace
    >>> import numpy as np

    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)

    >>> trA = trace(asvoigt(A))
    >>> trA
    3.3

    >>> np.allclose(trA, np.trace(A))
    True

    """

    return np.sum(A[:3], axis=0)


def transpose(A):
    r"""The major-transpose of a symmetric fourth-order tensor in reduced vector
    storage.

    Parameters
    ----------
    A : np.ndarray
        Symmetric fourth-order tensor in reduced vector storage.

    Returns
    -------
    np.ndarray
        Major-transpose of a symmetric fourth-order tensor in reduced vector storage.

    Notes
    -----

    ..  math::

        \mathbb{A}_{ijkl}^T = \mathbb{A}_{klij}


    Examples
    --------
    >>> from hyperelastic.math import asvoigt, dya, transpose
    >>> import numpy as np

    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> B = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> C = dya(asvoigt(A), asvoigt(B))
    >>> C
    array([[1.2 , 1.1 , 1.  , 1.4 , 1.3 , 1.5 ],
           [1.32, 1.21, 1.1 , 1.54, 1.43, 1.65],
           [1.44, 1.32, 1.2 , 1.68, 1.56, 1.8 ],
           [1.56, 1.43, 1.3 , 1.82, 1.69, 1.95],
           [1.68, 1.54, 1.4 , 1.96, 1.82, 2.1 ],
           [1.8 , 1.65, 1.5 , 2.1 , 1.95, 2.25]])

    >>> transpose(C)
    array([[1.2 , 1.32, 1.44, 1.56, 1.68, 1.8 ],
           [1.1 , 1.21, 1.32, 1.43, 1.54, 1.65],
           [1.  , 1.1 , 1.2 , 1.3 , 1.4 , 1.5 ],
           [1.4 , 1.54, 1.68, 1.82, 1.96, 2.1 ],
           [1.3 , 1.43, 1.56, 1.69, 1.82, 1.95],
           [1.5 , 1.65, 1.8 , 1.95, 2.1 , 2.25]])

    >>> D = astensor(C, mode=4)
    >>> E = astensor(transpose(C), mode=4)
    >>> np.allclose(D, np.einsum("klij", E))
    True

    """

    return np.swapaxes(A, 0, 1)


def det(A):
    """The determinant of a symmetric second-order tensor in reduced vector storage.

    Parameters
    ----------
    A : np.ndarray
        Symmetric second-order tensor in reduced vector storage.

    Returns
    -------
    np.ndarray
        Determinant of a symmetric second-order tensor in reduced vector storage.

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, det
    >>> import numpy as np

    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> B = asvoigt(A)

    >>> det(B)
    0.3169999999999993

    >>> np.allclose(det(B), np.linalg.det(A))
    True

    """

    return (
        A[0] * A[1] * A[2]
        + 2 * A[3] * A[4] * A[5]
        - A[4] ** 2 * A[0]
        - A[5] ** 2 * A[1]
        - A[3] ** 2 * A[2]
    )


def inv(A, determinant=None, out=None):
    r"""The inverse of a symmetric second-order tensor in reduced vector storage.

    Parameters
    ----------
    A : np.ndarray
        Symmetric second-order tensor in reduced vector storage.
    determinant : np.ndarray or None, optional
        The determinant of the symmetric second-order tensor (default is None).
    out : np.ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned (default is None).

    Returns
    -------
    np.ndarray
        Inverse of a symmetric second-order tensor in reduced vector storage.

    Notes
    -----

    ..  math::

        \boldsymbol{C} \boldsymbol{C}^{-1} = \boldsymbol{I}


    Examples
    --------
    >>> from hyperelastic.math import asvoigt, inv
    >>> import numpy as np

    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> B = asvoigt(A)

    >>> inv(B)
    array([-2.01892744, -3.31230284, -1.86119874,  1.70347003,  1.73501577, 0.5362776 ])

    >>> np.allclose(dot(B, inv(B)), np.eye(3))
    True

    """

    detAinvA = np.zeros_like(A)

    if determinant is None:
        detA = det(A)
    else:
        detA = determinant

    Ai = A[[1, 0, 0, 4, 3, 3]]
    Aj = A[[2, 2, 1, 5, 5, 4]]
    Ak = A[[4, 5, 3, 3, 0, 1]]
    Al = A[[4, 5, 3, 2, 4, 5]]

    if out is None:
        out = Ai

    Aij = np.multiply(Ai, Aj, out=Ai)
    Akl = np.multiply(Ak, Al, out=Ak)

    detAinvA = np.subtract(Aij, Akl, out=out)

    return np.divide(*np.broadcast_arrays(detAinvA, detA), out=out)


def dot(A, B, mode=(2, 2)):
    r"""The dot product of two symmetric second-order tensors in reduced vector storage,
    where the second index of the first tensor and the first index of the second tensor
    are contracted.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.
    mode : 2-tuple, optional
        The mode, 2 for second-order and 4 for fourth-order tensors (default is (2, 2)).

    Returns
    -------
    np.ndarray
        Dot product of two symmetric second-order tensors in reduced vector storage.

    Notes
    -----

    ..  math::

        C &= \boldsymbol{A} ~ \boldsymbol{B}

        C_{ij} &= A_{ik} B_{kj}

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, dot
    >>> import numpy as np

    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> B = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> C = dot(asvoigt(A), asvoigt(B), mode=(2, 2))
    >>> C
    array([[5.27, 4.78, 4.69],
           [5.2 , 4.85, 4.78],
           [5.56, 5.2 , 5.27]])

    >>> D = A @ B
    >>> np.allclose(C, D)
    True

    """

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


def eye(A=None):
    r"""A 3x3 tensor in Voigt-storage with ones on the diagonal and zeros elsewhere. The
    dimension is taken from the input argument (a symmetric second-order tensor in
    reduced vector storage).

    Parameters
    ----------
    A : np.ndarray or None, optional
        Symmetric second- or fourth-order tensor in reduced vector storage (default is
        None).

    Returns
    -------
    np.ndarray
        Identity matrix in reduced vector storage.

    Notes
    -----
    ..  math::

        \boldsymbol{I} = \begin{bmatrix} 1 & 1 & 1 & 0 & 0 & 0 \end{bmatrix}^T

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, eye
    >>> import numpy as np

    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> B = asvoigt(A)
    >>> eye(B)
    array([1., 1., 1., 0., 0., 0.])

    """

    trax = ()
    if A is not None:
        trax = np.ones(len(A.shape[1:]), dtype=int)

    return np.array([1, 1, 1, 0, 0, 0], dtype=float).reshape(6, *trax)


def ddot(A, B, mode=(2, 2)):
    r"""The double-dot product of two symmetric tensors in reduced vector storage, where
    the two innermost indices of both tensors are contracted.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second- or fourth-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second- or fourth-order tensor in reduced vector storage.
    mode : 2-tuple, optional
        The mode, 2 for second-order and 4 for fourth-order tensors (default is (2, 2)).

    Returns
    -------
    np.ndarray
        Double-dot product of two symmetric tensors in scalar or reduced vector/matrix
        storage.

    Notes
    -----

    ..  math::

        C &= \boldsymbol{A} : \boldsymbol{B}

        C &= A_{ij} : B_{ij}


    ..  math::

        \boldsymbol{C} &= \boldsymbol{A} : \mathbb{B}

        C_{kl} &= A_{ij} : \mathbb{B}_{ijkl}


    ..  math::

        \boldsymbol{C} &= \mathbb{B} : \boldsymbol{A}

        C_{ij} &= \mathbb{B}_{ijkl} : A_{kl}


    ..  math::

        \mathbb{C} &= \mathbb{A} : \mathbb{B}

        \mathbb{C}_{ijmn} &= \mathbb{A}_{ijkl} : \mathbb{A}_{klmn}


    Examples
    --------
    >>> from hyperelastic.math import asvoigt, ddot
    >>> import numpy as np

    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> B = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> A4 = np.einsum("ij,kl", A, B)
    >>> B4 = (np.einsum("ik,jl", A, A) + np.einsum("il,kj", A, A)) / 2

    >>> ddot(asvoigt(A), asvoigt(B), mode=(2, 2))
    15.39

    >>> ddot(asvoigt(A), asvoigt(B4, mode=4), mode=(2, 4))
    array([18.899, 18.863, 21.698, 18.903, 20.253, 20.316])

    >>> ddot(asvoigt(B4, mode=4), asvoigt(A), mode=(4, 2))
    array([18.899, 18.863, 21.698, 18.903, 20.253, 20.316])

    >>> ddot(asvoigt(A4, mode=4), asvoigt(B4, mode=4), mode=(4, 4))
    array([[18.519 , 18.787 , 21.944 , 18.675 , 20.326 , 20.225 ],
           [20.3709, 20.6657, 24.1384, 20.5425, 22.3586, 22.2475],
           [22.2228, 22.5444, 26.3328, 22.41  , 24.3912, 24.27  ],
           [24.0747, 24.4231, 28.5272, 24.2775, 26.4238, 26.2925],
           [25.9266, 26.3018, 30.7216, 26.145 , 28.4564, 28.315 ],
           [27.7785, 28.1805, 32.916 , 28.0125, 30.489 , 30.3375]])

    """

    weights = np.array([1, 1, 1, 2, 2, 2])
    if mode == (2, 2):
        return np.einsum("i...,i...,i->...", A, B, weights)
    elif mode == (4, 4):
        return np.einsum("ik...,kj...,k->ij...", A, B, weights)
    elif mode == (4, 2):
        return np.einsum("ij...,j...,j->i...", A, B, weights)
    elif mode == (2, 4):
        return np.einsum("i...,ij...,i->j...", A, B, weights)


def dya(A, B, out=None):
    r"""The dyadic product of two symmetric second-order tensors in reduced vector
    storage.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.
    out : np.ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned (default is None).

    Returns
    -------
    np.ndarray
        Dyadic product in reduced matrix storage.

    Notes
    -----
    The result of the dyadic product of two symmetric second order tensors
    is a minor- (but not major-) symmetric fourth-order tensor. For the case of
    :math:`\boldsymbol{A} = \boldsymbol{B}`, the result is both major- and minor-
    symmetric.

    ..  math::

        \mathbb{C} &= \boldsymbol{A} \otimes \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{ij}~B_{kl}

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, astensor, dya
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> A = np.einsum("ij,kl", C, D)

    >>> C6 = asvoigt(C, mode=2)
    >>> D6 = asvoigt(D, mode=2)

    >>> np.allclose(A, astensor(dya(C6, D6), mode=4))
    True

    """

    return np.multiply(
        A.reshape(-1, 1, *A.shape[1:]),
        B.reshape(1, -1, *B.shape[1:]),
        out=out,
    )


def dev(A):
    r"""The deviatoric part of a three-dimensional second-order tensor.

    Parameters
    ----------
    A : np.ndarray
        Symmetric second-order tensor in reduced vector storage.

    Returns
    -------
    np.ndarray
        Deviatoric part of the symmetric second-order tensor in reduced vector storage.

    Notes
    -----
    ..  math::

        \text{dev}(\boldsymbol{C}) =
         \boldsymbol{C} - \frac{\text{tr}(\boldsymbol{C})}{3} \boldsymbol{I}

    """
    return A - trace(A) / 3 * eye(A)


def cdya_ik(A, B, out=None):
    r"""The overlined-dyadic product of two symmetric second-order
    tensors in reduced vector storage, where the inner indices (the second index of the
    first tensor and the first index of the second tensor) are interchanged.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.
    out : np.ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned (default is None).

    Returns
    -------
    np.ndarray
        Overlined-dyadic product in full-array storage.

    Notes
    -----
    The result of the overlined-dyadic product of two symmetric second order tensors
    is a major- (but not minor-) symmetric fourth-order tensor. This is also the case
    for :math:`\boldsymbol{A} = \boldsymbol{B}`.

    ..  math::

        \mathbb{C} &= \boldsymbol{A} \overline{\otimes} \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{ik}~B_{jl}

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, cdya_ik
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> A = np.einsum("ik,jl", C, D)

    >>> C6 = asvoigt(C, mode=2)
    >>> D6 = asvoigt(D, mode=2)

    >>> np.allclose(A, cdya_ik(C6, D6))
    True

    """

    return np.swapaxes(astensor(dya(A, B, out=out), mode=4), 1, 2)


def cdya_il(A, B, out=None):
    r"""The underlined-dyadic product of two symmetric second-order
    tensors in reduced vector storage, where the right indices of the two tensors are
    interchanged.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.
    out : np.ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned (default is None).

    Returns
    -------
    np.ndarray
        Underlined-dyadic product in full-array storage.

    Notes
    -----
    The result of the underlined-dyadic product of two symmetric second
    order tensors is a non-symmetric fourth-order tensor. In case of
    :math:`\boldsymbol{A} = \boldsymbol{B}`, the fourth-order tensor is major-symmetric.

    ..  math::

        \mathbb{C} &= \boldsymbol{A} \underline{\otimes} \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{il}~B_{kj}

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, cdya_il
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> A = np.einsum("il,kj", C, D)

    >>> C6 = asvoigt(C, mode=2)
    >>> D6 = asvoigt(D, mode=2)

    >>> np.allclose(A, cdya_il(C6, D6))
    True

    """

    return np.swapaxes(astensor(dya(A, B, out=out), mode=4), 1, 3)


def cdya(A, B, out=None):
    r"""The full-symmetric crossed-dyadic product of two symmetric second-order tensors
    in reduced vector storage.

    Parameters
    ----------
    A : np.ndarray
        First symmetric second-order tensor in reduced vector storage.
    B : np.ndarray
        Second symmetric second-order tensor in reduced vector storage.
    out : np.ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned (default is None).

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
            \boldsymbol{A} \odot \boldsymbol{B} + \boldsymbol{B} \odot \boldsymbol{A}
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

    ..  note::
        The technical implementation is based on an answer of Jérôme Richard (see
        `stackoverflow <https://stackoverflow.com/questions/76640596>`_).

    Examples
    --------
    >>> from hyperelastic.math import asvoigt, astensor, cdya
    >>> import numpy as np

    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3)
    >>> D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2])[::-1].reshape(3, 3)

    >>> CD = (np.einsum("ik,jl", C, D) + np.einsum("il,kj", C, D)) / 2
    >>> DC = (np.einsum("ik,jl", D, C) + np.einsum("il,kj", D, C)) / 2

    >>> A = (CD + DC) / 2

    >>> C6 = asvoigt(C, mode=2)
    >>> D6 = asvoigt(D, mode=2)
    >>> cdya(C6, D6)
    array([[1.2   , 1.82  , 2.25  , 1.48  , 2.025 , 1.65  ],
           [1.82  , 1.21  , 1.82  , 1.485 , 1.485 , 1.825 ],
           [2.25  , 1.82  , 1.2   , 2.025 , 1.48  , 1.65  ],
           [1.48  , 1.485 , 2.025 , 1.515 , 1.7375, 1.7575],
           [2.025 , 1.485 , 1.48  , 1.7375, 1.515 , 1.7575],
           [1.65  , 1.825 , 1.65  , 1.7575, 1.7575, 1.735 ]])

    >>> np.allclose(A, astensor(cdya(C6, D6), mode=4))
    True

    >>> np.allclose(A, astensor(cdya(D6, C6), mode=4))
    True

    """

    i, j = [a.ravel() for a in np.triu_indices(6)]

    a = np.array([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)])
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    ij = b[i, j]
    kl = b[k, l]

    ik = b[i, k]
    jl = b[j, l]

    il = b[i, l]
    kj = b[k, j]

    if out is None:
        out = np.zeros((6, *np.broadcast_shapes(A.shape, B.shape)))

    A, B = np.broadcast_arrays(A, B)
    Aik, Bjl, Ail, Bkj = A[ik], B[jl], A[il], B[kj]

    values = np.multiply(Aik, Bjl, out=Aik) + np.multiply(Ail, Bkj, out=Ail)

    if A is not B:
        Bik, Ajl, Bil, Akj = B[ik], A[jl], B[il], A[kj]

        values += np.multiply(Bik, Ajl, out=Bik) + np.multiply(Bil, Akj, out=Bil)
        values /= 4
    else:
        values /= 2

    out[ij, kl] = out[kl, ij] = values

    return out


def eigh(A, fun=None):
    "Eigenvalues and -bases of matrix A."
    wA, vA = np.linalg.eigh(astensor(A).T)
    MA = np.einsum("...ia,...ja->...ija", vA, vA)
    if fun is not None:
        wA = fun(wA)
    return wA.T, np.array([asvoigt(M) for M in MA.T])
