module QuteeCUDAExt
using CUDA
using OMEinsum
using LinearAlgebra

import Qutee: QuantumInfo
import Qutee: Optimization
import Qutee: Random

"""
    rand_channel(::Type{CuArray}, r, n)

"""
function QuantumInfo.rand_channel(::Type{CuArray}, r, n)
    K = CUDA.randn(ComplexF32, (r * n, n))
    K = reshape(CuArray(qr(K).Q), (r, n, n))
    return CuArray(permutedims(K, (2, 3, 1)))
end

function Optimization.qr_retraction(G::CuMatrix{T}, K::CuMatrix{T}, η::Float32=0.01f0) where {T}
    # Compute the perturbed point
    G = G / norm(G, 2)
    Y = K - η * G

    # Perform QR decomposition
    q, r = qr(Y)
    d = diag(r)
    D = Diagonal(sign.(d .+ 1 // 2))

    return CuArray(q) * D
end

"""
    optimize(K::CuArray{T,3}, f, iter::Int32=Int32(100); retraction=qr_retraction, η::Float32=0.1f0, decay_factor::Float32=0.5f0, decay_step::Int32=Int32(10)) where {T}
"""
function Optimization.optimize(K::CuArray{T}, f, iter::Int32=Int32(100); retraction=qr_retraction, η::Float32=0.1f0, decay_factor::Float32=0.5f0, decay_step::Int32=Int32(10)) where {T}
    history = [f(K)]
    @progress for i in 1:iter
        push!(history, f(K))
        grad = Zygote.gradient(x -> f(x), K)[1]
        K = retraction(kraus2stiefel(grad), kraus2stiefel(K), η)
        K = stiefel2kraus(K)
        # Apply step decay every decay_step iterations
        if i % decay_step == 0
            η *= decay_factor
        end
    end
    return K, history
end

function Random.ginibre_matrix(::Type{CuArray}, d, k)
    return CUDA.randn(ComplexF64, (d, k)) + im * CUDA.randn(ComplexF64, (d, k))
end

@info("Qutee loaded the CUDA module successfully")
end