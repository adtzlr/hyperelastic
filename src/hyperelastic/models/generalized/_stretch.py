def strain(stretch, exponent):
    "Seth-Hill strain as a function of the stretch (with first and second derivatives)."

    k = exponent
    λ = stretch

    E = (λ**k - 1) / k
    dEdλ = λ ** (k - 1)
    d2Edλdλ = (k - 1) * λ ** (k - 2)

    d2Edλdλ0 = 3

    return E, dEdλ, d2Edλdλ, d2Edλdλ0


def deformation(stretch, exponent):
    """Generalized deformation as a function of the stretch (with first and second
    derivatives)."""

    k = exponent
    λ = stretch

    Ck = λ**k
    dCkdλ = k * λ ** (k - 1)
    d2Ckdλdλ = k * (k - 1) * λ ** (k - 2)

    d2Cdλdλ0 = 6

    return Ck, dCkdλ, d2Ckdλdλ, d2Cdλdλ0
