using Test
using Qutee
using CUDA

@test true

# @test begin
#     # Reset two qubits
#     R = reshape([1.0 0 0 1; 0 0 0 0], (2, 2, 2))
#     R ⊗ R ≈ [1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
# end

# @test begin
#     R = reshape([1. 0 0 1; 0 0 0 0], (2,2,2)) # reset gate
#     M = reshape([1 0 0 0; 0 0 0 1], (2,2,2)) # measurement
#     # Combined reset and measurement is the same as just a reset 
#     R⋅M ≈ R && M⋅R≈R
# end

@test begin
    L = QuantumInfo.rand_channel(CuArray, 2,2)
    x = CUDA.randn(2,2)
    L * x
end