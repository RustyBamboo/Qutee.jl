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


## Analysis
The following code uses the goodness function to assess how effective certain methods are at generating eigenvectors. We will use it to test how well the arnoldi method performs.

```julia

using LinearAlgebra, Pkg, CUDA
using Qutee

nn = 4  # number of qubits
cases = 10 # the number of test cases
comp = zeros((2^nn)^2,1)
for j in 1:cases
	En = QuantumInfo.rand_channel(Array, 3, 2^nn)  # Random channel in Kraus representation
	Ln = QuantumInfo.kraus2liou(En)  # Convert channel to a big matrix representation
	goal = rand((2^nn)^2)  # Random start vector for arnoldi

	vals, standeig = standardeigen(Ln)  # True eigenvalues to compare to
	for i in 1:(2^nn)^2
		testvals, testeig = arnoldi2eigen(Ln,goal,(2^nn)^2,i)  # For each iteration, find arnoldi eigenvalues
		comp[i,1] += goodness(standeig,testeig,i)  # Store number of matches
	end
end	
comp /= cases  # averages the results
for i in 1:(2^nn)^2  # prints result for easy copy + paste to store for later
	print(comp[i,1])
	print(", ")
end

```

We can plot the data generated via this process. Here, I use the data I got running the arnoldi method and averaging over 10 tests. Running more than 4 qubits made my computer die :(

```julia
using LinearAlgebra, Pkg, Plots, BenchmarkTools, CUDA
using Qutee

# ind is for the x axis (number of iterations m) 
ind1 = zeros((2^1)^2,1)
for i in 1:(2^1)^2
	ind1[i,1] = i
end
ind2 = zeros((2^2)^2,1)
for i in 1:(2^2)^2
	ind2[i,1] = i
end
ind3 = zeros((2^3)^2,1)
for i in 1:(2^3)^2
	ind3[i,1] = i
end
ind4 = zeros((2^4)^2,1)
for i in 1:(2^4)^2
	ind4[i,1] = i
end

# Values I got from my tests
comp1 = [0.0, 0.2, 0.8, 3.2]
comp2 = [0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.9, 1.0, 1.0, 1.4, 1.6, 2.0, 2.9, 3.5, 5.5, 10.2]
comp3 = [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.2, 1.3, 1.4, 1.4, 1.4, 1.5, 1.6, 1.7, 2.1, 2.3, 2.1, 2.4, 2.6, 2.4, 3.1, 3.0, 3.3, 3.5, 4.2, 3.4, 4.8, 4.9, 5.1, 4.9, 5.8, 5.9, 5.7, 6.4, 7.1, 6.4, 8.5, 7.7, 9.1, 8.8, 8.0, 9.8, 9.5, 11.0, 10.1, 10.6, 12.3, 12.4, 14.8, 14.9, 22.9, 34.0]
comp4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.3, 1.4, 1.5, 1.5, 1.7, 1.9, 2.0, 2.0, 1.9, 2.2, 2.2, 2.8, 2.4, 2.9, 2.8, 3.0, 2.7, 2.5, 2.7, 2.3, 2.5, 2.8, 2.7, 2.7, 3.8, 3.3, 2.9, 2.7, 2.9, 3.1, 3.0, 2.6, 3.3, 3.0, 3.5, 3.0, 3.6, 4.1, 4.0, 4.3, 3.9, 5.1, 5.1, 4.7, 4.8, 5.4, 5.3, 5.8, 5.8, 5.6, 5.5, 5.6, 5.5, 6.3, 6.1, 5.1, 6.7, 7.8, 7.3, 7.6, 7.0, 8.5, 8.3, 7.6, 7.8, 7.6, 8.1, 8.0, 7.5, 7.9, 7.5, 7.4, 7.4, 8.6, 9.7, 8.8, 10.0, 11.3, 10.7, 11.0, 11.9, 12.2, 12.1, 12.7, 11.7, 12.4, 12.9, 12.8, 13.0, 13.2, 14.1, 12.4, 12.2, 12.2, 12.2, 11.0, 12.0, 12.1, 11.9, 13.1, 14.0, 15.3, 16.4, 15.0, 14.8, 15.7, 14.7, 15.4, 16.2, 13.0, 15.5, 15.3, 16.4, 17.4, 19.0, 17.2, 14.8, 16.0, 18.0, 17.5, 17.0, 15.9, 16.3, 19.2, 16.3, 16.9, 18.2, 17.6, 17.3, 17.4, 18.9, 20.2, 21.2, 21.4, 22.5, 23.0, 21.0, 22.0, 24.3, 22.6, 21.4, 24.6, 24.3, 21.0, 23.2, 26.1, 23.7, 26.8, 25.2, 22.9, 25.4, 25.1, 25.7, 26.5, 27.9, 25.2, 28.0, 27.8, 29.0, 31.6, 31.2, 30.5, 30.7, 30.8, 31.0, 30.9, 32.8, 30.7, 32.5, 35.4, 32.5, 31.8, 34.5, 37.4, 34.7, 37.6, 33.8, 35.8, 36.0, 37.0, 39.4, 39.3, 38.9, 39.8, 37.9, 40.8, 41.5, 43.4, 44.3, 41.3, 42.1, 45.4, 44.2, 42.0, 46.1, 45.6, 51.9, 51.8, 56.8, 53.9, 47.9, 50.5, 57.3, 57.8, 54.5, 55.7, 60.7, 60.6, 61.1, 59.8, 93.4, 131.4]

# Plotting the results
plot(ind4[1:100,1],comp4[1:100,1],label="n=4")
plot!(ind3[1:50,1],comp3[1:50,1],label="n=3")
plot!(ind2,comp2,label="n=2")
plot!(ind1,comp1,label="n=1")
```