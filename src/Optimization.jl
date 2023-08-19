module Optimization

using CUDA
using LinearAlgebra, Zygote, ProgressLogging

export kraus2stiefel
export stiefel2kraus
export optimize

"""
    Preforms the retraction step for Manifold optimization

    G is the gradient
    K is the point
"""
function retraction(G::Matrix{T}, K::Matrix{T}, η=0.1) where {T}
    G = G / norm(G, 2)
    A = hcat(G, K)
    B = hcat(K, -G)

    prod = B' * A
    mid = inv(I(size(prod, 1)) + η * prod / 2)
    grad = A * (mid) * B' * K
    return K - η * grad
end

function qr_retraction(G::Matrix{T}, K::Matrix{T}, η=0.01) where {T}
    # Compute the perturbed point
    Y = K - η * G

    # Perform QR decomposition
    q, r = qr(Y)
    d = diag(r)
    D = Diagonal(sign.(sign.(d .+ 1 // 2)))

    return Array(q) * D
end

function qr_retraction(G::CuMatrix{T}, K::CuMatrix{T}, η=0.01f0) where {T}
    # Compute the perturbed point
    Y = CuArray(K - η * G)

    # Perform QR decomposition
    q, r = qr(Y)
    d = diag(r)
    D = Diagonal(sign.(sign.(d .+ 1 // 2)))

    return CuArray(q) * D
end

"""
    optimize(K::Array{T,3}, f, iter=100; retraction=retraction, η=0.1, decay_factor=0.5, decay_step=10) where {T}

    A simple gradient descent algorithm along manifold
"""
function optimize(K::Array{T,3}, f, iter=100; retraction=retraction, η=0.1, decay_factor=0.5, decay_step=10) where {T}
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

"""
    optimize(K::CuArray{T,3}, f, iter::Int32=Int32(100); retraction=qr_retraction, η::Float32=0.1f0, decay_factor::Float32=0.5f0, decay_step::Int32=Int32(10)) where {T}
"""
function optimize(K::CuArray{T}, f, iter::Int32=Int32(100); retraction=qr_retraction, η::Float32=0.1f0, decay_factor::Float32=0.5f0, decay_step::Int32=Int32(10)) where {T}
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


"""
    Optimization done over batches of Stiefel elements.

    Good for optimization circuits consisting of a sequence of unitary gates
"""
function optimize(K::Vector{T}, f, iter=100; η=0.1, decay_factor=0.5, decay_step=10) where {T}
    history = [f(K)]
    @progress for i in 1:iter
        push!(history, f(K))
        grad = Zygote.gradient(x -> f(x), K)[1]
        K = retraction.(kraus2stiefel.(grad), kraus2stiefel.(K), η)
        K = stiefel2kraus.(K)
        # Apply step decay every decay_step iterations
        if i % decay_step == 0
            η *= decay_factor
        end
    end
    return K, history
end


kraus2stiefel(K) = vcat([K[:, :, i] for i in 1:size(K, 3)]...)

function stiefel2kraus(k)
    rn, n = size(k)
    r = rn ÷ n
    return permutedims(reshape(transpose(k), (n, n, r)), [2, 1, 3])
end

end