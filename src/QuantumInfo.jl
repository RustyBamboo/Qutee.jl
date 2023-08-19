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

function kraus2liou(K::Array{T,3}) where {T}
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

"""
    Provides a random quantum channel as Kraus operators

    r = rank (max should be n^2)
    n = size
"""
function rand_channel(r, n)
    K = randn(ComplexF64, (r * n, n))
    K = reshape(Matrix(qr(K).Q), (r, n, n))
    return permutedims(K, (2, 3, 1))
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
    return λ, v₀ / sum(diag(v₀))
end

end