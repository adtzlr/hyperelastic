def strain(stretch, exponent):
    "Seth-Hill strain as a function of the stretch (with first and second derivatives)."

    k = exponent
    λ = stretch

    E = lambda λ, k: (λ**k - 1) / k  # noqa: E731
    dEdλ = lambda λ, k: λ ** (k - 1)  # noqa: E731
    d2Edλdλ = lambda λ, k: (k - 1) * λ ** (k - 2)  # noqa: E731

    E0 = E(λ=1, k=2)
    dEdλ0 = dEdλ(λ=1, k=2)
    d2Edλdλ0 = d2Edλdλ(λ=1, k=2)

    return E(λ, k), dEdλ(λ, k), d2Edλdλ(λ, k), E0, dEdλ0, d2Edλdλ0


def deformation(stretch, exponent):
    """Generalized deformation as a function of the stretch (with first and second
    derivatives)."""

    k = exponent
    λ = stretch

    Ck = lambda λ, k: λ**k  # noqa: E731
    dCkdλ = lambda λ, k: k * λ ** (k - 1)  # noqa: E731
    d2Ckdλdλ = lambda λ, k: k * (k - 1) * λ ** (k - 2)  # noqa: E731

    C0 = Ck(λ=1, k=2)
    dCdλ0 = dCkdλ(λ=1, k=2)
    d2Cdλdλ0 = d2Ckdλdλ(λ=1, k=2)

    return Ck(λ, k), dCkdλ(λ, k), d2Ckdλdλ(λ, k), C0, dCdλ0, d2Cdλdλ0
