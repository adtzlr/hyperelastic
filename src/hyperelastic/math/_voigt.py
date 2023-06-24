import numpy as np


def asvoigt(A, mode=2):
    """Convert 3x3 tensor to symmetric (Voigt-notation) vector storage of shape 6."""

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
    "Convert symmetric 6 tensor in Voigt-notation to full 3x3 tensor."
    if mode == 2:  # second order tensor of shape 6 to 3x3
        a = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)
        return A[a]

    elif mode == 4:  # fourth order tensor of shape 6x6 to 3x3x3x3
        a = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2])
        i, j = np.meshgrid(a, a)
        return A[i.reshape(3, 3, 3, 3), j.reshape(3, 3, 3, 3)]


def tril_from_triu(A, inplace=True):
    "Copy upper triangle values to lower triangle values of a 3x3 tensor inplace."
    B = A
    if not inplace:
        B = A.copy()

    i, j = np.triu_indices(6, 1)
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
        C[0] = A[0] * B[0] + A[3] * B[3] + A[5] * B[5]
        C[1] = A[3] * B[3] + A[1] * B[1] + A[4] * B[4]
        C[2] = A[4] * B[4] + A[5] * B[5] + A[2] * B[2]
        C[3] = A[0] * B[3] + A[3] * B[1] + A[5] * B[4]
        C[4] = A[0] * B[5] + A[3] * B[4] + A[5] * B[2]
        C[5] = A[3] * B[5] + A[1] * B[4] + A[4] * B[2]

    return C


def eye(A):
    "A 3x3 tensor in Voigt-storage with ones on the diagonal and zeros elsewhere."
    ntrax = len(A.shape[1:])
    return np.array([1, 1, 1, 0, 0, 0]).reshape(6, *np.ones(ntrax, dtype=int))


def ddot(A, B, mode=(2, 2)):
    "The double-contraction of two symmetric 3x3 tensors in Voigt-storage."
    ntrax = len(A.shape[1:])
    if mode == (2, 2):
        weights = np.array([1, 1, 1, 2, 2, 2]).reshape(6, *np.ones(ntrax, dtype=int))
        return np.sum(A * B * weights, axis=0)
    elif mode == (4, 4):
        weights = np.array([1, 1, 1, 2, 2, 2])
        return np.einsum("ik...,kj...,k->ij...", A, B, weights)


def dya(A, B):
    "The dyadic (outer) product."
    return A.reshape(-1, 1, *A.shape[1:]) * B.reshape(1, -1, *B.shape[1:])


def dev(A):
    "The deviatoric part of a 3x3 tensor."
    return A - trace(A) / 3 * eye(A)


def cdya_ik(A, B):
    "The inner-crossed dyadic (outer) product."
    i, j = [a.ravel() for a in np.indices((9, 9))]

    a = np.array(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    )
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    m = b[i, k].reshape(3, 3, 3, 3)
    n = b[j, l].reshape(3, 3, 3, 3)

    return A[m] * B[n]


def cdya_il(A, B):
    "The outer-crossed dyadic (outer) product."
    i, j = [a.ravel() for a in np.indices((9, 9))]

    a = np.array(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    )
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    m = b[i, l].reshape(3, 3, 3, 3)
    n = b[k, j].reshape(3, 3, 3, 3)

    return A[m] * B[n]


def cdya(A, B):
    "The inner- and outer-crossed dyadoc (outer) product."
    i, j = [a.ravel() for a in np.indices((6, 6))]

    a = np.array([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)])
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    m = b[i, k].reshape(6, 6)
    n = b[j, l].reshape(6, 6)

    r = b[i, l].reshape(6, 6)
    s = b[k, j].reshape(6, 6)

    return (A[m] * B[n] + A[r] * A[s]) / 2
