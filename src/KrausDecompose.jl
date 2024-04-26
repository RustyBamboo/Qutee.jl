module KrausDecompose

using LinearAlgebra

struct KrausRepresentation
    U::AbstractMatrix
    rho_d::AbstractMatrix
    kraus_ops::Dict{Tuple{Int64,Int64},AbstractMatrix}
    function KrausRepresentation(U, rho_d)
        kraus_ops = compute_kraus(U, rho_d)
        new(U, rho_d, kraus_ops)
    end
end

function compute_kraus(U::AbstractMatrix, rho_d::AbstractMatrix)

    dim_c = size(U)[1]
    dim_d = size(rho_d)[1]
    dim_s = dim_c ÷ dim_d

    p_d, ψ_d = eigen(Hermitian(rho_d))

    kraus_ops = Dict{Tuple{Int64,Int64},AbstractMatrix}()

    for i in 1:dim_d
        ψ_i = ψ_d[:, i]

        if p_d[i] ≈ 0
            continue
        end

        for k in 1:dim_d
            ψ_k = zeros(dim_d, 1)
            ψ_k[k] = 1

            left = kron(ψ_k', I(dim_s))
            right = kron(ψ_i,   I(dim_s))

            kraus = sqrt(p_d[i]) * left * U * right
            kraus_ops[(k - 1, i - 1)] = kraus
        end
    end
    return kraus_ops
end

function Base.Array(self::KrausRepresentation)
    return stack(values(self.kraus_ops))
end

function Base.display(self::KrausRepresentation)
    return display(self.kraus_ops)
end

end