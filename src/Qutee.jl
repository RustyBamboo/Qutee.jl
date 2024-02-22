"""
    The QUantum Toolbox
"""
module Qutee
using OMEinsum

include("QuantumInfo.jl")
export QuantumInfo

export ⊗
export ⊙

function Base.kron(a::Array{T,3}, b::Array{S,2}) where {T,S}
    j, k, n = size(a)
    l, m = size(b)

    @ein c[m, k, l, j, n] := a[j, k, n] * b[l, m]
    c = permutedims(reshape(c, (j * l, k * m, n)), (2, 1, 3))
    return c
end

function Base.kron(b::Array{T,2}, a::Array{S,3}) where {T,S}
    j, k, n = size(a)
    l, m = size(b)

    @ein c[m, k, l, j, n] := b[j, k] * a[l, m, n]
    c = permutedims(reshape(c, (j * l, k * m, n)), (2, 1, 3))
    return c
end

function Base.kron(a::Array{T,3}, b::Array{S,3}) where {T,S}
    j, k, n = size(a)
    l, m = size(b)

    @ein c[m, k, l, j, n] := a[j, k, n] * b[l, m, n]
    c = permutedims(reshape(c, (j * l, k * m, n)), (2, 1, 3))
    return c
end

⊗ = kron

"""
    apply(ρ, K) = ∑KᵢρKᵢ

    Applies an operation to a density matrix.
"""
function apply(ρ::AbstractArray{T,2}, op::AbstractArray{S,3}) where {T,S}
    op_conj = conj(op)
    @ein c[i, k, n] := ρ[i, j] * op_conj[k, j, n]
    @ein d[i, k] := op[i, j, n] * c[j, k, n]

    return d
    # return sum(op .* reshape(ρ,(size(ρ)...,1)) .* conj(permutedims(op, (2,1,3))), dims=3)
end

function apply(ρ::AbstractArray{T,2}, op::AbstractArray{S,2}) where {T,S}
    return apply(ρ, reshape(op, (size(op)..., 1)))
end

function Base.:*(K::AbstractArray{T,3}, v::AbstractArray{S,2}) where {T,S}
    return apply(v, K)
end

function Base.adjoint(K::AbstractArray{T,3}) where {T}
    K = conj(permutedims(K, (2, 1, 3)))
    return K
end

# function Base.adjoint(K::AbstractArray{T,3}, v::AbstractArray{S,2}) where {T,S}
# K = conj(permutedims(K, (2, 1, 3)))
# return apply(v, K)
# end

"""
    dot(K,L)

    Combine two processes in the Kraus representation
"""
function dot(a::Array{T,3}, b::Array{S,3}) where {T,S}
    @ein c[i, k, n, m] := a[i, j, n] * b[j, k, m]
    c = reshape(c, (size(a, 2), size(b, 1), :))

    # Filter out zero ops
    # zero_slices = mapslices(x -> all(x .≈ 0), c, dims=[1, 2])
    # indices = findall(!, zero_slices)
    # linear_indices = [i[3] for i in indices]

    # return c[:, :, linear_indices]
    return c
end

function dot(a::Array{T,2}, b::Array{S,3}) where {T,S}
    return dot(reshape(a, size(a)..., 1), b)
end

function dot(a::Array{T,3}, b::Array{S,2}) where {T,S}
    return dot(a, reshape(b, size(b)..., 1))
end

function dot(a::Array{T,2}, b::Array{S,2}) where {T,S}
    return a * b
end

⊙ = dot


function p_tr(K::Array{T,3}, n, m) where {T}

    j, k, i = size(K)
    K_tensor = reshape(K, (n, m, n, m, i))
    @ein K_ptr[j, k, i] := K_tensor[f, j, f, k, i]
    return K_ptr
end

end # module Qutee
