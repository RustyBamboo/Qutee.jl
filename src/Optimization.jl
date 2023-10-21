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

"""
    qr_retraction(G::Matrix{T}, K::Matrix{T}, η=0.01) where {T}

    Perform QR-based retraction. Note that the performance is worse, and requires picking suitable hyperparmeters.
"""
function qr_retraction(G::Matrix{T}, K::Matrix{T}, η=0.01) where {T}
    # Compute the perturbed point
    G = G / norm(G, 2)
    Y = K - η * G

    # Perform QR decomposition
    q, r = qr(Y)
    d = diag(r)
    D = Diagonal(sign.(sign.(d .+ 1 // 2)))

    return Array(q) * D
end

function qr_retraction(G::CuMatrix{T}, K::CuMatrix{T}, η::Float32=0.01f0) where {T}
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
    The following function is adapted from the paper "Efficient Riemannian Optimization On The Stiefel Manifold Via The Cayley Transform"
        by Jun Li, Li Fuxin, Sinisa Todorovic

    optimize(X::Array{T,3}, f, epochs, M, v, l, β_1, β_2, d, d_time, 𝜀, q, s) where {T}

    Cayley-Adam optimization along a Stiefel manifold.

    X is the initial point on the Stiefel matrix manifold we are optimizing.
    f is the loss function.
    
    epochs is the number of epochs.
    M is the initial value of the first momentum. Its size should match that of X.
    v is the initial vlaue of the second momentum.
    l is the learning rate. In this case, we use the learning rate along the Stiefel manifold.
    β_1 is the first momentum coefficient.
    β_2 is the second momentum coefficient.
    d is the decay factor of the learning rate.
    d_time is the epochs at which the decay factor should be applied.
    𝜀 is a small value used to avoid division by 0.
    q is a coefficient used when deciding the step size.
    s is the number of iterations used to estimate the Cayley Transform

    
    Returns the matrix X after optimization, and a history of the state of X after each epoch.
"""
function optimize_adam(X::AbstractArray{T,3}, f, epochs=100, M=fill!(similar(X), zero(T)), v=1, l=0.4, β_1=0.9, β_2=0.99, d=0.2, d_time=[30, 60, 120, 160], 𝜀=10^-8, q=0.5, s=2) where {T}
    M = kraus2stiefel(M)  # Reshaping the first momentum to its Stiefel form. Allows for operations later.
    history = [f(X)]  # An array of the state of X at every epoch.
    @progress for k in 1:epochs  # Iterates for each epoch
        push!(history, f(X))  # Adds the current state of X to the history
        # Decays learning rate if appropriate
        if k in d_time
            l *= (1 - d)
        end
        grad = Zygote.gradient(x -> f(x), X)[1]  # Obtains gradient function (takes the Kraus form of X)
        grad = kraus2stiefel(grad)  # Reshapes the gradient function to its Stiefel form. Allows for operations later.
        X = kraus2stiefel(X)  # Converts the matrix X to its Stiefel form. Allows for operations later.

        M = β_1 * M + (1 - β_1) * grad  # Estimate biased momentum
        v = β_2 * v + (1 - β_2) * norm(grad)^2
        v_n = v / (1 - β_2^k)  # Update biased second raw moment estimate
        r = (1 - β_1^k) * sqrt(v_n + 𝜀)  # Estimate biased-corrected ratio
        W = M * X' - 1 / 2 * X * (X' * M * X')  # Compute the auxillary skew-symmetric matrix
        W = (W - W') / r
        M = r * W * X  # Project momentum onto the tangent space
        a = min(l, 2q / (norm(W) + 𝜀))  # Select adaptive learning rate for contraction mapping
        Y = X - a * M  # Iterative estimation of the Cayley Transform
        for i in 1:s
            Y = X - a / 2 * W * (X + Y)
        end
        X = Y
        X = stiefel2kraus(X)  # Converts X back to Kraus form so gradient can be found in next iteration.
    end
    return X, history
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
