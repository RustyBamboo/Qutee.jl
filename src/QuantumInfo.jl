module QuantumInfo
using LinearAlgebra, TensorOperations, CUDA, cuTENSOR

include("Optimization.jl")
export Optimization

mat(v::Vector, r=round(Int, sqrt(length(v))), c=round(Int, sqrt(length(v)))) = reshape(v, r, c)

R = reshape([1.0 0 0 1; 0 0 0 0], (2, 2, 2)) # reset gate
M = reshape([1 0 0 0; 0 0 0 1], (2, 2, 2)) # measurement

"""
The Liovulle and Choi representation is related via an involution

Dividing Quantum Channels
Michael M. Wolf, J. Ignacio Cirac
"""
function choi_liou_involution(r::Matrix)
    d = round(Int, sqrt(size(r, 1)))
    rl = reshape(r, (d, d, d, d))
    rl = permutedims(rl, [1, 3, 2, 4])
    reshape(rl, size(r))
end

function liou2choi(r::Matrix)
    choi_liou_involution(r)
end

function choi2liou(r::Matrix)
    choi_liou_involution(r)
end

function kraus2liou(K::AbstractArray{T,3}) where {T}
    j, k, i = size(K)

    K_conj = conj(K)

    @tensor S[j, l, k, m] := K[j, k, i] * K_conj[l, m, i]

    return reshape(S, (k * j, k * j))
end

function choi2kraus(r::Matrix{T}) where {T}
    d = sqrt(size(r, 1))
    (vals, vecs) = eigen(d * r)
    #vals = eigvals( sqrt(size(r,1))*r )
    kraus_ops = Matrix{T}[]
    for i in eachindex(vals)
        push!(kraus_ops, sqrt(round(vals[i], digits=15, RoundToZero)) * mat(vecs[:, i]))
    end
    factor = tr(sum([k' * k for k in kraus_ops]))
    kraus_ops = kraus_ops / sqrt(factor / d)
    return cat(kraus_ops..., dims=3)
end


"""
  Takes a liouville superoperator (computational basis) and
  returns a vector of the equivalen Krawu vectors
"""
function liou2kraus(l::Matrix)
    choi2kraus(liou2choi(l))
end

const PAULIS = [
    [1 0im; 0 1],
    [0im 1; 1 0],
    [0 -im; im 0],
    [1 0im; 0 -1]
]

function num2quat(n::Int, l::Int)::Vector{Int}
    return map(s -> parse(Int, s), collect(string(n, base=4, pad=l)))
end

function toPauli(p::Vector{Int})::Matrix{Complex{T}} where {T}
    return reduce(kron, [PAULIS[x+1] for x in p])
end

"""
    liou2pauliliou(m::Matrix{T})

Converts a Liouville superoperator in the computational basis to one in the Pauli basis.
Order of Paulis is I,X,Y and Z.

# Arguments
- `m`: Matrix to convert.

# Returns
- Converted matrix.
"""
function liou2pauliliou(m::Matrix{T})::Matrix{Float64} where {T}
    if size(m, 1) != size(m, 2)
        error("Only square matrices supported")
    elseif size(m, 1) != 4^(floor(log2(size(m, 1)) / 2))
        error("Only matrices with dimension 4^n supported.")
    end

    dsq = size(m, 1)
    res = zeros(ComplexF64, dsq, dsq)
    l = round(Int, log2(dsq) / 2)
    pauliVectors = [vec(toPauli(num2quat(i - 1, l))) for i = 1:dsq]
    pauliVectorDaggers = [x' for x in pauliVectors]
    normalization = sqrt(dsq)

    for i = 1:dsq
        for j = 1:dsq
            res[i, j] = pauliVectorDaggers[i] * m * pauliVectors[j] / normalization
        end
    end

    return real(res)
end

"""
    rand_channel(::Type{Array}, r, n)

    Provides a random quantum channel as Kraus operators

    r = rank (max should be n^2)
    n = size
"""
function rand_channel(::Type{Array}, r, n)
    K = randn(ComplexF64, (r * n, n))
    K = reshape(Matrix(qr(K).Q), (r, n, n))
    return permutedims(K, (2, 3, 1))
end

"""
    rand_channel(::Type{CuArray}, r, n)

"""
function rand_channel(::Type{CuArray}, r, n)
    # K = randn(ComplexF32, (r * n, n)) |> cu
    K = CUDA.randn(ComplexF32, (r * n, n))
    K = reshape(CuArray(qr(K).Q), (r, n, n))
    return CuArray(permutedims(K, (2, 3, 1)))
end


function fidelity(ρ::Matrix, σ::Matrix)
    # Calculate the square root of ρ
    eigenρ = eigen(ρ)
    sqrtρ = eigenρ.vectors * Diagonal(sqrt.(eigenρ.values)) * eigenρ.vectors'

    # Compute the product sqrt(ρ) σ sqrt(ρ)
    temp = sqrtρ * σ * sqrtρ

    # Calculate the trace of the square root of the result
    eigentemp = eigen(temp)
    result = tr(Diagonal(sqrt.(clamp.(real.(eigentemp.values), 0, Inf))))  # clamp to ensure non-negative values

    # Return the squared result
    return abs(result)^2
end


function qubit_density(rhos)
    """
        Get the coefficients for Pauli matrices of a single qubit density matrix
    """
    purity_s = Float64[]
    a_s = Float64[]
    b_s = Float64[]
    c_s = Float64[]

    for m in rhos
        a = 2 * real(m[2, 1])
        b = 2 * imag(m[2, 1])
        c = 2 * real(m[1, 1]) - 1

        push!(purity_s, abs(tr(m * m)))
        push!(a_s, a)
        push!(b_s, b)
        push!(c_s, real(c))
    end

    return purity_s, a_s, b_s, c_s
end

"""
    power_method(A, v₀, max_iterations=1000, tol=1e-6)

    Uses the power method to compute the largest eigenvalue and eigenvector.
    Note: the implementation is not optimized, however it works with automatic differentiation
    Note: the returned vector may not be a physical density matrix (you may need to divide by tr(out))
"""
function power_method(A, v₀, max_iterations=1000, tol=1e-6)
    # w = similar(v₀)
    for _ = 1:max_iterations
        w = A * v₀
        v_new = w / norm(w)

        # Check convergence
        if norm(v_new - v₀) < tol
            break
        end
        v₀ = v_new
    end

    λ = dot(v₀, A * v₀)  # Rayleigh quotient
    return λ, v₀
end

"""
    arnoldi(A,b,n,m)
    Thanks Laurent Hoeltgen's blog for explaining how Krylov subspaces work: https://laurenthoeltgen.name/project/num-linalg-opt/

    Uses the arnoldi method to generate the Hessenberg matrix from A, an n x n matrix. 
    b is a n x 1 vector that we use as our initial guess.
    m is the number of iterations we do, following a Gram-Schmidt like process.

    Outpts q, a n x m matrix where the m^th column is an n x 1 vector representing the m^th iteration.
    h is the Hessenberg matrix, which is "almost" triagonal, and is tridiagonal when A is symmetric.
"""
function arnoldi(A, b, n, m)
    # initializes Q and H
    q = zeros(ComplexF64, n, m)
    q[:, 1] = b / norm(b)  # sets q1, the first column of Q, as our initial guess
    h = zeros(ComplexF64, m, m)
    for j in 1:m-1  # does m-1 iterations
        t = A * q[:, j]
        for i in 1:j
            h[i, j] = q[:, i]' * t   # H[i,j] = qi' * t * qj
            t = t - h[i, j] * q[:, i]  # Orthogonalizes
        end
        h[j+1, j] = norm(t)
        q[:, j+1] = (t / h[j+1, j])  # finds q for the next iteration
    end
    h[:, m] = q' * (A * q[:, m])   # Finishes the final iteration
    return (q, h)
end


"""
    arnoldi2eigen(A,b,n,m)

    Finds the eigenvalues and eigenvectors of A, an n x n matrix by using the Arnoldi method.
    b is an n x 1 vector that serves as our initial guess for the arnoldi method
    m is the number of iterations we perform (related to how many eigenvectors we want)

    In the arnoldi method, H has the same eigenvalues as A!
    If we call the eigenvectors of H, v. Then, Qv will give us the eigenvectors of A!

    returns ev, an Array of the eigenvalues, and yeek, an n x m Array of the m eigenvectors, each size n x 1 
"""
function arnoldi2eigen(A, b, n, m)
    q, h = arnoldi(A, b, n, m)  # Finds Q and H from the arnoldi method
    ev = eigvals(h)  # Eigenvalues of H, same as for A 
    eek = eigvecs(h)  # Eigenvectors of H, can be modified to produce eigenvectors of A
    yeek = zeros(ComplexF64, n, m)
    for i in 1:m
        yeek[:, i] = q * eek[:, i]  # Qv = eigenvectors of A
    end
    return (ev, yeek)
end

end