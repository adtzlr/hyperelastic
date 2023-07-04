import numpy as np

import hyperelastic
import hyperelastic.math as hm


def test_math():
    np.random.seed(546123)
    F = np.eye(3).reshape(3, 3, 1, 1) + np.random.rand(3, 3, 4, 10) / 10
    C = np.einsum("ki...,kj...->ij...", F, F)
    A = np.einsum("ij...,kl...->ijkl...", C, C)
    C6 = hm.asvoigt(C, mode=2)
    A6 = hm.asvoigt(A, mode=4)

    assert np.allclose(hm.tril_from_triu(A6, inplace=False), A6)
    assert np.allclose(hm.triu_from_tril(A6, inplace=False), A6)

    assert np.allclose(hm.inv(C6), hm.inv(C6, determinant=hm.det(C6)))
    assert np.allclose(hm.dot(C6, C6), np.einsum("ik...,kj...->ij...", C, C))

    assert np.allclose(hm.ddot(C6, A6, mode=(2, 4)), hm.ddot(A6, C6, mode=(4, 2)))

    assert np.allclose(hm.cdya_il(C6, C6), np.einsum("il...,kj...->ijkl...", C, C))
    assert np.allclose(hm.cdya_ik(C6, C6), np.einsum("ik...,jl...->ijkl...", C, C))


if __name__ == "__main__":
    test_math()
