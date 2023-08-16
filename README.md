# Qutee.jl

> [!WARNING]  
> This is research software. Not all features are implemented or optimized. 

This code aims to achieve the following goals:
- Construction of quantum circuits as a quantum channel
- Riemannian optimization for quantum channel (circuit) optimization
- Conversion of quantum information forms (Kraus, Choi, Liouville, etc.)
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
rand_U = [reshape(QuantumInfo.rand_channel(1,2^2), (2^2,2^2)) for _ in 1:3]

# Construction of the circuit
function circuit(U)
	C = mapreduce(x->op⊙x, ⊙, U)
	return C
end

# The loss function that we wish to optimize
function circuit_error(U)
    ...
end

# Optimize using gradient descent over the 
out_u, history_u = QuantumInfo.Optimization.optimize(rand_U, circuit_error, 500; η=0.2, decay_factor=0.9, decay_step=10)

```