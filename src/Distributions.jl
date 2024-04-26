module Distributions
using Random
using Distributions

±(a, b) = [a - b, a + b]

"""
    MarchenkoPastur(N,d)
    
"""
struct MarchenkoPastur <: ContinuousUnivariateDistribution
    N::Float64
    d::Float64
    λ::Float64
    function MarchenkoPastur(N=2.0, d=4.0, λ=1)
        new(N, d, λ)
    end
end

function Distributions.pdf(d::MarchenkoPastur, x::Real)
    κ = 1 / (d.N * d.d)
    if d.λ > 1
        if x < 0
            return 0
        end
        if x == 0
            return 1 - 1 / d.λ
        end
    end

    λ₋, λ₊ = κ * (1 ± sqrt(d.d)) .^ 2

    if x < λ₋ || x > λ₊
        return 0
    else
        return 1 / (2π * κ) * (sqrt((λ₊ - x) * (x - λ₋))) / (d.λ * x)
    end
end

function Base.rand(rng::AbstractRNG, d::MarchenkoPastur)
    κ = 1 / (d.N * d.d)

    λ₋, λ₊ = κ * (1 ± sqrt(d.d)) .^ 2

    while true
        x = rand(Uniform(λ₋, λ₊))
        if rand() <= pdf(d, x)
            return x
        end
    end
end




end