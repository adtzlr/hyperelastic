import numpy as np


def as_voigt(A):
    return A[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]


def as_tensor(A, mode=2):
    trax = A.shape[1:]
    return A[[0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(3, 3, *trax)


def tril_from_triu(A):
    a, b = np.triu_indices(6, 1)
    for i, j in zip(a, b):
        A[j, i] = A[i, j]
    return A


def trace(A):
    return np.sum(A[:3], axis=0)


def det(A):
    return (
        A[0] * A[1] * A[2]
        + 2 * A[3] * A[5] * A[4]
        - A[4] ** 2 * A[0]
        - A[5] ** 2 * A[1]
        - A[2] * A[3] ** 2
    )


def inv(A, determinant=None):
    detAinvA = np.zeros_like(A)

    if determinant is None:
        detA = det(A)
    else:
        detA = determinant

    detAinvA[0] = -A[3] ** 2 + A[1] * A[2]
    detAinvA[1] = -A[4] ** 2 + A[0] * A[2]
    detAinvA[2] = -A[3] ** 2 + A[0] * A[1]

    detAinvA[3] = A[4] * A[5] - A[3] * A[2]
    detAinvA[4] = -A[3] * A[1] + A[3] * A[5]
    detAinvA[5] = A[4] * A[3] - A[0] * A[5]

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


def ddot(A, B):
    ntrax = len(A.shape[1:])
    C = np.array([1, 1, 1, 2, 2, 2]).reshape(6, *np.ones(ntrax, dtype=int))
    return np.sum(A * B * C, axis=0)


def dya(A, B):
    trax = A.shape[1:]
    return A.reshape(-1, 1, *trax) * B.reshape(1, -1, *trax)


def dev(A):
    return A - trace(A) / 3 * eye(A)


def cdya_ik(A, B):
    trax = A.shape[1:]
    C = np.zeros((6, 6, *trax))
    for i in range(3):
        C[i, i] = A[i] * B[i]

    C[3, 4] = A[0] * B[5]

    C[0, 1] = A[3] * B[3]
    C[0, 2] = A[4] * B[4]
    C[1, 2] = A[5] * B[5]

    C[1, 3] = A[3] * B[1]
    C[4, 5] = A[3] * B[2]
    C[0, 5] = A[3] * B[4]
    C[1, 4] = A[3] * B[5]

    C[2, 4] = A[4] * B[2]
    C[2, 3] = A[4] * B[5]

    C[2, 5] = A[5] * B[2]

    return tril_from_triu(C)
