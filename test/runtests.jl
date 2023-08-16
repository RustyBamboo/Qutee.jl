using Test
using Qutee

@test true

@test begin
    # Reset two qubits
    R = reshape([1.0 0 0 1; 0 0 0 0], (2, 2, 2))
    R ⊗ R ≈ [1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
end

@test begin
    R = reshape([1. 0 0 1; 0 0 0 0], (2,2,2)) # reset gate
    M = reshape([1 0 0 0; 0 0 0 1], (2,2,2)) # measurement
    # Combined reset and measurement is the same as just a reset 
    R⋅M ≈ R && M⋅R≈R
end