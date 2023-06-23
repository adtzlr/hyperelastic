import felupe.math as fm
import numpy as np


def as_voigt(A, mode=2):
    if mode == 2:
        i = [0, 1, 2, 0, 1, 0]
        j = [0, 1, 2, 1, 2, 2]

        return A[i, j]

    elif mode == 4:
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


def as_tensor(A, mode=2):
    if mode == 2:
        a = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)
        return A[a]
    elif mode == 4:
        a = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2])
        i, j = np.meshgrid(a, a)
        return A[i.reshape(3, 3, 3, 3), j.reshape(3, 3, 3, 3)]


def tril_from_triu(A):
    i, j = np.triu_indices(6, 1)
    A[j, i] = A[i, j]
    return A


def trace(A):
    return np.sum(A[:3], axis=0)


def transpose(A):
    return np.einsum("ij...->ji...", A)


def det(A):
    return (
        A[0] * A[1] * A[2]
        + 2 * A[3] * A[4] * A[5]
        - A[4] ** 2 * A[0]
        - A[5] ** 2 * A[1]
        - A[3] ** 2 * A[2]
    )


def inv(A, determinant=None):
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
    if mode == (2, 2):
        trax = np.broadcast_shapes(A.shape[1:], B.shape[1:])
        C = np.zeros((3, 3, *trax))
        C[0] = A[0] * B[0] + A[3] * B[3] + A[4] * B[4]
        C[1] = A[3] * B[3] + A[1] * B[1] + A[5] * B[5]
        C[2] = A[4] * B[4] + A[5] * B[5] + A[2] * B[2]
        C[3] = A[0] * B[3] + A[3] * B[1] + A[4] * B[5]
        C[4] = A[0] * B[4] + A[3] * B[5] + A[4] * B[2]
        C[5] = A[3] * B[4] + A[1] * B[5] + A[5] * B[2]

    return C


def eye(A):
    ntrax = len(A.shape[1:])
    return np.array([1, 1, 1, 0, 0, 0]).reshape(6, *np.ones(ntrax, dtype=int))


def ddot(A, B, mode=(2, 2)):
    ntrax = len(A.shape[1:])
    if mode == (2, 2):
        C = np.array([1, 1, 1, 2, 2, 2]).reshape(6, *np.ones(ntrax, dtype=int))
        return np.sum(A * B * C, axis=0)
    elif mode == (4, 4):
        trax = np.broadcast_shapes(A.shape[2:], B.shape[2:])
        C = np.zeros((6, 6, *trax))
        for i in range(6):
            for k in range(6):
                for j in range(6):
                    if k >= 3:
                        w = 2
                    else:
                        w = 1
                    C[i, j] += A[i, k] * B[k, j] * w
        return C


def dya(A, B):
    return A.reshape(-1, 1, *A.shape[1:]) * B.reshape(1, -1, *B.shape[1:])


def dev(A):
    return A - trace(A) / 3 * eye(A)


def cdya_ik(A, B):
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
    i, j = [a.ravel() for a in np.indices((6, 6))]

    a = np.array([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)])
    b = np.array([0, 3, 5, 3, 1, 4, 5, 4, 2]).reshape(3, 3)

    i, j, k, l = np.hstack([a[i], a[j]]).T

    m = b[i, k].reshape(6, 6)
    n = b[j, l].reshape(6, 6)

    r = b[i, l].reshape(6, 6)
    s = b[k, j].reshape(6, 6)

    return (A[m] * B[n] + A[r] * A[s]) / 2

    # trax = np.broadcast_shapes(A.shape[1:], B.shape[1:])
    # C = np.zeros((6, 6, *trax))

    # C[0, 0] = A[0] * B[0]

    # C[0, 1] = A[3] * B[3]
    # C[1, 1] = A[1] * B[1]

    # C[0, 2] = A[5] * B[5]
    # C[1, 2] = A[4] * B[4]
    # C[2, 2] = A[2] * B[2]

    # C[0, 3] = (A[0] * B[3] + A[3] * B[0]) / 2
    # C[1, 3] = (A[3] * B[1] + A[1] * B[3]) / 2
    # C[2, 3] = (A[5] * B[4] + A[4] * B[5]) / 2
    # C[3, 3] = (A[0] * B[1] + A[3] * B[3]) / 2

    # C[0, 4] = (A[5] * B[3] + A[3] * B[5]) / 2
    # C[1, 4] = (A[4] * B[1] + A[1] * B[4]) / 2
    # C[2, 4] = (A[2] * B[4] + A[4] * B[2]) / 2
    # C[3, 4] = (A[3] * B[4] + A[5] * B[1]) / 2
    # C[4, 4] = (A[1] * B[2] + A[4] * B[4]) / 2

    # C[0, 5] = (A[0] * B[5] + A[5] * B[0]) / 2
    # C[1, 5] = (A[3] * B[4] + A[4] * B[3]) / 2
    # C[2, 5] = (A[5] * B[2] + A[2] * B[5]) / 2
    # C[3, 5] = (A[5] * B[3] + A[0] * B[4]) / 2
    # C[4, 5] = (A[4] * B[5] + A[3] * B[2]) / 2
    # C[5, 5] = (A[2] * B[0] + A[5] * B[5]) / 2

    # return tril_from_triu(C)


def piola(F, C4):
    FF = as_voigt(fm.cdya(F, F) + fm.cdya(fm.transpose(F), fm.transpose(F)), mode=4) / 2
    return ddot(FF, C4, mode=(4, 4))
