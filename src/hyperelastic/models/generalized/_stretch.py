def strain(stretch, exponent):
    "Seth-Hill strain as a function of the stretch (with first and second derivatives)."

    k = exponent
    λ = stretch

    E = (λ**k - 1) / k
    dEdλ = λ ** (k - 1)
    d2Edλdλ = (k - 1) * λ ** (k - 2)

    return E, dEdλ, d2Edλdλ


def deformation(stretch, exponent):
    """Generalized deformation as a function of the stretch (with first and second
    derivatives)."""

    k = exponent
    λ = stretch

    Ck = 2 / k * λ**k
    dCkdλ = 2 * λ ** (k - 1)
    d2Ckdλdλ = 2 * (k - 1) * λ ** (k - 2)

    return Ck, dCkdλ, d2Ckdλdλ
