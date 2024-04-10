module QuteeAMDGPUExt
using Qutee
import Qutee: QuantumInfo

using AMDGPU
using LinearAlgebra


"""
    rand_channel(::Type{ROCArray}, r, n)

"""
function QuantumInfo.rand_channel(::Type{ROCArray}, r, n)
    K = AMDGPU.randn(Float32, (r * n, n)) + AMDGPU.randn(Float32, (r * n, n))
    K = reshape(ROCArray(qr(0.5*K).Q), (r, n, n))
    return ROCArray(permutedims(K, (2, 3, 1)))
end

@info("Qutee loaded the AMDGPU module successfully")
end