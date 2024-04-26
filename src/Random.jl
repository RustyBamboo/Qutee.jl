module Random
using LinearAlgebra, OMEinsum

function ginibre_matrix(::Type{Array}, d, k)
    return randn(ComplexF64, (d, k)) + im * randn(ComplexF64, (d, k))
end


"""
    rand_channel(::Type{Array}, r, n)

Create a random Choi matrix of a Hilbert space dimension n with a kraus rank r.
Returns a n^2 by n^2 Choi matrix.

Random quantum operations.
    Bruzda et al.
    Physics Letters A 373, 320 (2009).
    https://doi.org/10.1016/j.physleta.2008.11.043
    https://arxiv.org/abs/0804.2361
"""
function rand_channel(::Type{Array}, r, n)
    X = ginibre_matrix(Array, n^2, r)
    rho = X * X'

    rho_r = reshape(rho, (n, n, n, n))
    @ein rho_r[j, k] := rho_r[i, j, i, k]

    Q = kron(sqrt(inv(rho_r)), I(n))
    Z = Q * rho * Q
    return Z
end

"""
    rand_density_hs(::Type{Array}, n, r)

Create a random channel from the Hilbert-Shmidt metric
"""
function rand_density_hs(::Type{Array}, n, r)
    X = ginibre_matrix(Array, n, r)
    rho = X * X'
    return rho / tr(rho)
end


"""
    rand_density_bures(::Type{Array}, n, r)

Create a random channel from the Bures metric
"""
function rand_density_bures(::Type{Array}, n, r)

    X = ginibre_matrix(Array, n, n) / sqrt(2)
    f = qr(X)

    rho = I(n) + Matrix(f.Q)
    rho = rho * ginibre_matrix(Array, n, r)
    rho = X * X'
    return rho / tr(rho)
end

end