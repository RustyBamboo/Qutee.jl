# Qutee.jl

> [!WARNING]  
> This is research software. Not all features are implemented or optimized. 

This code aims to achieve the following goals:
- Construction of quantum circuits as a quantum channel
- Riemannian optimization for quantum channel (circuit) optimization
- Conversion of quantum information forms (Kraus, Choi, Liouville, etc.)
- Automatic differentiation (AD)
- GPU support (with AD)
- Someday use [Enzyme](https://github.com/EnzymeAD/Enzyme) for AD
- And more?

## Installation

For now, this package is not registered. But you can use directly from Github. 

```julia
using Pkg
Pkg.add(url="https://github.com/RustyBamboo/Qutee.jl")
```
or
```julia
]
pkg> add https://github.com/RustyBamboo/Qutee.jl
```

## Examples

**Circuit Optimization**

```julia
using Qutee

# 2-qubit operation: Qubit reset and Identity
op = (QuantumInfo.R ⊗ [1 0; 0 1]) 

# A vector of random 2-qubit gates (4x4 unitary matrices)
# Note that in the rand_channel I had to pass as Array as my first parameter to specify what type of channel I want
rand_U = [reshape(QuantumInfo.rand_channel(Array,1,2^2), (2^2,2^2)) for _ in 1:3]

# Construction of the circuit
function circuit(U)
	C = mapreduce(x->op⊙x, ⊙, U)
	return C
end

# The loss function that we wish to optimize. You can compare to whatever matrix you want.
# For reference, here I use a simple example where we compare to the identity matrix.
function circuit_error(U)
    C = circuit(U)  # Creates a circuit from our matrix
	C_L = QuantumInfo.kraus2liou(C)  # Converts the circuit to a form we can work with
	return norm(C_L - I)  # Example loss function
end

# Optimize using gradient descent over the Stiefel Manifold
out_u, history_u = QuantumInfo.Optimization.optimize(rand_U, circuit_error, 500; η=0.2, decay_factor=0.9, decay_step=10)

```

**CUDA**

A small benchmark that compares CPU and CUDA in
1. Matrix-matrix multiplication
2. Finding the largest eigenpair via the [power method](https://en.wikipedia.org/wiki/Power_iteration) 

```julia
using LinearAlgebra, Pkg, Plots, BenchmarkTools, CUDA
using Qutee

function random_vector(K)
	n,m,_ = size(K)
	v = rand(n,m) + im * rand(n,m)
    v /= norm(v)
end

# Note that I passed a CuArray as the first paramter in rand_channel because I want to create a CUDA channel
K_list = [QuantumInfo.rand_channel(CuArray,2,2^i) for i in 1:11]
v_list = [random_vector(K) for K in K_list]

cpu_times = [@elapsed K_list[i] * v_list[i] for i in 1:length(K_list)]
gpu_times = [CUDA.@elapsed cu(K_list[i]) * cu(v_list[i]) for i in 1:length(K_list)]
p1 = plot([cpu_times, gpu_times], labels=["CPU" "CUDA"], title="Matrix-Matrix Multiplication", xlabel="# of Qubits", ylabel="Time [s]", markershape=:xcross)
savefig(p1, "benchmark1.png")

cpu_times_power = [@elapsed QuantumInfo.power_method(K_list[i], v_list[i], 50) for i in 1:length(K_list)]
gpu_times_power = [@elapsed QuantumInfo.power_method(cu(K_list[i]), cu(v_list[i]), 50) for i in 1:length(K_list)]
p2 = plot([cpu_times_power, gpu_times_power], labels=["CPU" "CUDA"], title="Power Method (50 Iterations)", xlabel="# of Qubits", ylabel="Time [s]", markershape=:xcross)
savefig(p2, "benchmark2.png")
```

![](docs/src/gfx/benchmark1.png)
![](docs/src/gfx/benchmark2.png)
