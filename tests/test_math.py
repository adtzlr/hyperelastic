import numpy as np

import hyperelastic.math as hm


def test_math():
    np.random.seed(546123)
    F = np.eye(3).reshape(3, 3, 1, 1) + np.random.rand(3, 3, 4, 10) / 10
    C = np.einsum("ki...,kj...->ij...", F, F)
    D = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3, 1, 1) * C
    A = np.einsum("ij...,kl...->ijkl...", C, C)
    C6 = hm.asvoigt(C, mode=2)
    D6 = hm.asvoigt(D, mode=2)
    A6 = hm.asvoigt(A, mode=4)

    assert np.allclose(hm.tril_from_triu(A6), A6)
    assert np.allclose(hm.triu_from_tril(A6), A6)
    assert np.allclose(hm.trace(hm.dev(C6)), 0)

    assert np.allclose(hm.inv(C6), hm.inv(C6, determinant=hm.det(C6)))
    assert np.allclose(hm.dot(C6, C6), np.einsum("ik...,kj...->ij...", C, C))

    assert np.allclose(hm.ddot(C6, A6, mode=(2, 4)), hm.ddot(A6, C6, mode=(4, 2)))

    assert np.allclose(hm.cdya_il(C6, C6), np.einsum("il...,kj...->ijkl...", C, C))
    assert np.allclose(hm.cdya_ik(C6, C6), np.einsum("ik...,jl...->ijkl...", C, C))

    CD = (
        np.einsum("ik...,jl...->ijkl...", C, D)
        + np.einsum("il...,kj...->ijkl...", C, D)
    ) / 2
    DC = (
        np.einsum("ik...,jl...->ijkl...", D, C)
        + np.einsum("il...,kj...->ijkl...", D, C)
    ) / 2
    B = (CD + DC) / 2

    assert np.allclose(B, hm.astensor(hm.cdya(C6, D6), mode=4))
    assert np.allclose(B, hm.astensor(hm.cdya(D6, C6), mode=4))


if __name__ == "__main__":
    test_math()
