### A Pluto.jl notebook ###
# v0.20.0

#> [frontmatter]

using Markdown
using InteractiveUtils

# ╔═╡ e9fbbd5d-6cea-4e1c-807b-6dc3939d3628
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".")

# ╔═╡ 92f32d6f-de53-4794-8451-5a3ca62ef828
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ b9970d7f-9a69-424e-89c0-bbfcfe5f12d7
using OMEinsum

# ╔═╡ c3cc1bd0-68e6-430c-b306-696c51165b9c
using Yao

# ╔═╡ 332085f3-ef12-4a7d-9714-e8f4a88d5a28
using GenericTensorNetworks

# ╔═╡ 5c3dfeeb-3dd2-4a4d-a1b7-2d292beccf73
using LuxorGraphPlot

# ╔═╡ 1845d970-4f90-4714-bc74-6e38b27160f4
using TensorInference

# ╔═╡ ca6bee1f-d3d3-4620-aba9-363e5b856c69
md"""
# Tensor network ecosystem in Julia
"""

# ╔═╡ a481f737-c83a-4f55-b1b5-cbb33ddbc9f4
md"""
# Large scale tensor network and its contraction order
"""

# ╔═╡ c12e790f-769c-4277-b41d-81d08c0e134c
md"""
## OMEinsum.jl

A tensor network contraction engine featuring:
- Hyper-optimized contraction order
- Automatic differentiation and GPU support
"""

# ╔═╡ ec23abba-eaf0-4e7a-bf7e-b50ffbe9532f
md"""
## Matrix multiplication in tensor network representation
"""

# ╔═╡ 0a83fd07-6dd3-4228-b435-c7f0c1ddbc12
code = ein"ij,jk->ik"

# ╔═╡ e82c8140-c80e-4683-a72a-c8dec034c0a9
getixsv(code)

# ╔═╡ 8f153a7f-3977-44ce-8c9c-7540f6c84e2c
getiyv(code)

# ╔═╡ 4e849834-1f00-4283-ad7c-f7a783d77ab4
A, B = randn(3,3), randn(3, 3);

# ╔═╡ 0ddf1061-d53c-4561-943b-f4460ee58962
# einsum code can be called liked a function
R = code(A, B)

# ╔═╡ c4e8f71a-3118-4455-a9a5-b427e2d19ba0
md"""
Meaning
```julia
for i = 1:n_i
    for j = 1:n_j
        for k = 1:n_k
        	R[i, k] += A[i, j] * B[j, k]
		end
	end
end
```
"""

# ╔═╡ 6af3c70e-5a7e-4ab8-baf1-d666fea321cb
md"""
## Contracting multiple tensors
"""

# ╔═╡ 3b9821b6-851d-4509-aa7c-7ac7c8d57399
code_long = ein"ij,jk,kl,lm->im"

# ╔═╡ 56cfb6d5-1866-4318-a578-7b30660b1f45
show_einsum(code_long; layout=StressLayout(optimal_distance=15))

# ╔═╡ cee4366b-a6f0-4a75-bcab-82b77b31b451
C, D = randn(3, 3), randn(3, 3)

# ╔═╡ 15c74390-bb16-4ea6-8e6e-70f9628304c4
code_long(A, B, C, D)

# ╔═╡ 008b26e8-862b-4e4a-8d76-fd671defca4f
md"""
## The matrix product state in physics
"""

# ╔═╡ 27c769f8-52ec-496c-bea7-cc210af9df99
code_mps = ein"aj,jbk,kcl,ldm,me->abcde"

# ╔═╡ f93ed0d4-bb49-462e-88ad-3995c2a9532c
tensors = [randn(2,20), randn(20,2,40), randn(40,2,40), randn(40,2,20), randn(20,2)];

# ╔═╡ ca5d4d5b-dfe3-4fd8-a5b2-6cbe70ef84a4
code_mps(tensors...) |> size

# ╔═╡ 6884f655-aac8-4f22-8de7-e2624f1e9246
show_einsum(code_mps; layout=StressLayout(optimal_distance=10))

# ╔═╡ 4d412abe-17b3-46bf-99fe-b133b20610ee
md"""
## The difference between `einsum` and Einstein's notation
"""

# ╔═╡ 3c423eb5-b64a-4256-a41c-7d675d7396c8
# In eincode, a variable can be shared by more than 2 tensors
code_withbatch = ein"ijb,jkb->ikb"

# ╔═╡ 6ce21263-bf2c-41b1-ac1f-5b4b29c94465
md"## Optimize the contraction order"

# ╔═╡ c4eac832-3629-41bf-8dc9-5b2a6d3d914b
size_dict = OMEinsum.get_size_dict(getixsv(code_mps), tensors)

# ╔═╡ 3c84890c-0371-44cb-a9c7-ae76adeb0307
# The complexity is too high
contraction_complexity(code_mps, size_dict)

# ╔═╡ 98cafa86-1fa5-43b9-b479-3f0893fecbb8
md"""
- Time complexity is the number of "`*`" operations.
- Space complexity is the maximum size of intermediate tensors.
- Read-write complexity is the number of element read and write.
"""

# ╔═╡ f80aa1e7-1a92-4a52-a070-d382fa36dcb3
optcode = optimize_code(code_mps, size_dict, TreeSA())

# ╔═╡ b4d09576-c8b3-4363-bd45-1364e4918862
md"TODO: explain slicing"

# ╔═╡ 750ab594-9990-4ad1-a8c8-e66ea058909d
contraction_complexity(optcode, size_dict)

# ╔═╡ 9411e8b2-aa55-487f-b666-5eb1e3a821d9
md"## The available optimization methods"

# ╔═╡ dce59830-0885-40fb-9693-b9f070f2388b
subtypes(CodeOptimizer)

# ╔═╡ 63e793e3-6eb1-4c65-8041-99beb33dc1fa
md"""
- GreedyMethod: fast but not optimal
- ExactTreewidth: optimal but exponential time
- TreeSA: heuristic local search, close to optimal, slicing supported
- KaHyParBipartite and SABipartite: min-cut based bipartition, better heuristic for extremely large tensor networks
"""

# ╔═╡ 1484dee3-df3f-4aaa-8c32-78eea3133e8f
md"""
## Theoretical optimal contraction order

*Theorem*: The optimal space complexity of tensor network contraction is equal to the *tree width* of its *line graph* $L(G)$.
"""

# ╔═╡ 3fc75abf-5f97-4534-87aa-7c5944e58d64
md"![](https://arrogantgao.github.io/assets/treewidth_figs/treedecomposition_square.png)"

# ╔═╡ 90f342ba-0a56-4702-afd5-6a8c49964b19
md"""
Please check out the blog: Finding the Optimal Tree Decomposition with Minimal Treewidth, [https://arrogantgao.github.io/blogs/treewidth/](https://arrogantgao.github.io/blogs/treewidth/)

And also Xuan-Zhao Gao's talk tomorrow.
"""

# ╔═╡ 25f8e2f2-022e-4c65-aff3-4af8fe43a5dc
md"""
# Tensor network for quantum circuit simulation: Yao.jl
"""

# ╔═╡ 33f37f98-53e9-45b4-919a-2b9b1481f4ce
md"""
## Simulate A quantum circuit
"""

# ╔═╡ f32a5030-8c21-4859-82e8-8953cf8b534f
function qft_circuit(n::Int)
    c = chain(n)
    for i = 1:n
        push!(c, put(n, i=>H))
        for j = i+1:n
            push!(c, control(n, j, i => shift(π/2^(j-i))))
        end
    end
    return c
end

# ╔═╡ 7b87b462-4c32-40f7-ae8f-f68df3d42e45
qft = qft_circuit(4);

# ╔═╡ 10242d23-2a97-43d5-874f-db283dfbd274
vizcircuit(qft)

# ╔═╡ f4e57c8d-57d3-4ee1-9cfe-06bfac8ac87f
mat(control(2, 1, 2=>shift(Basic(π)/2)))

# ╔═╡ 49dfaa9b-bcbc-4f60-8496-34561e57d098
md"""
## Fron gates to tensors
"""

# ╔═╡ d96500ba-ade4-4666-a37b-222673960117
md"The Hadamard gate"

# ╔═╡ 5956ee36-d0b8-4231-a5ae-b18d5c02842e
vizcircuit(H)

# ╔═╡ ddb5868a-7406-4f6f-971d-16394f4cea3b
mat(Basic, H)  # Basic is the symbolic type

# ╔═╡ 9671c446-775e-4cb3-8e95-7c32480eeb07
md"The tensor network representation:"

# ╔═╡ a3f038a5-8f09-4c33-bf71-ed07e03037b3
nodestore() do ns
	dx = 30
	r = 10
	sr = 3
	dots = [dot!(0, 0), dot!(2dx, 0)]
	c = circle!((dx, 0), r)
	cons = [Connection(dots[1], c), Connection(c, dots[2])]

	with_nodes(ns; padding_left=20, padding_right=20) do
		stroke(c)
		LuxorGraphPlot.text("H", c)
		stroke.(cons)
		LuxorGraphPlot.text("i", -10, 0)
		LuxorGraphPlot.text("j", 2dx+10, 0)
	end
end

# ╔═╡ 5af0aca3-acd3-4172-b922-91c5568fbb35
md"""
The CPhase gate
"""

# ╔═╡ d336e16b-4af8-4564-9205-3ae51e61c33e
vizcircuit(control(2, 2, 1=>shift(Basic(:θ))))

# ╔═╡ b7735171-7345-49da-916e-b8c9942eec84
md"""
The tensor network representation
"""

# ╔═╡ 6173ea8a-a9e7-40e8-badb-601f8e96bd7d
nodestore() do ns
	dx = 30
	r = 10
	sr = 3
	dy = 40
	dots = [dot!(0, 0), dot!(2dx, 0), dot!(0, dy), dot!(2dx, dy)]
	c = circle!((dx, 0.5dy), r)
	scs = [circle!((dx, 0), sr), circle!((dx, dy), sr)]
	cons = [Connection(dots[1], dots[2]), Connection(dots[3], dots[4]), Connection(scs[1], c), Connection(scs[2], c)]

	with_nodes(ns; padding_top=20, padding_bottom=20) do
		stroke(c)
		fill.(scs)
		LuxorGraphPlot.text("θ", c)
		stroke.(cons)
		LuxorGraphPlot.text("i", dx, -8)
		LuxorGraphPlot.text("j", dx, dy+15)
	end
end

# ╔═╡ cb012a72-f5d8-41b7-9146-0cb184269801
reshape(ein"ij->ijij"([1 1; 1 exp(im*Basic(π)/2)]), 4, 4)

# ╔═╡ d2833db0-7f86-493b-b96e-8747f782f129
md"## Convert a circuit to a tensor network"

# ╔═╡ 29a41d88-7531-4914-866f-bc4185bde8c2
vizcircuit(qft)

# ╔═╡ ae35e03a-0da9-496f-b942-ff312fb2b366
md"""
The inputs state $|0\rangle$ is represented as:
"""

# ╔═╡ faafeff1-c964-440c-a2e8-640fd323f5ae
nodestore() do ns
	dx = 30
	r = 10
	sr = 3
	dots = [dot!(dx, 0)]
	c = circle!((0, 0), sr)
	cons = [Connection(dots[1], c)]

	with_nodes(ns; padding_left=20, padding_right=20) do
		stroke(c)
		stroke.(cons)
		LuxorGraphPlot.text("i", dx+10, 0)
	end
end

# ╔═╡ a991e849-b397-40c6-8036-b9aa728f4c22
md"""
The output is: 
"""

# ╔═╡ e229dd53-bf34-4048-8464-61cc35225225
nodestore() do ns
	n = 4
	dx = 40
	x0 = 20
	y0 = 20
	r = 10
	sr = 3
	dy = 40
	cs = []
	dots = []
	inputs = []
	outputs = []
	hs = []
	cons = []
	for j = 1:n
		push!(inputs, circle!((x0, y0+j*dy), sr))  # inputs
	end
	for i = 1:n
		x0 += dx
		push!(hs, circle!((x0, y0+i*dy), r))
		push!(cons, Connection(inputs[i], hs[i]))
		for j = i+1:n
			x0 += dx
			push!(cs, circle!((x0, y0+(j - 0.5) * dy), r))
			push!(dots, circle!((x0, y0+j*dy), sr))
			push!(dots, circle!((x0, y0+(j-1)*dy), sr))
			push!(cons, Connection(dots[end], cs[end]))
			push!(cons, Connection(dots[end-1], cs[end]))
		end
	end
	for j=1:n
		push!(outputs, circle!((x0+dx, y0+j*dy), sr))  # outputs
		push!(cons, Connection(outputs[j], hs[j]))
	end
	with_nodes(ns) do
		stroke.(cs)
		LuxorGraphPlot.text.(["π/2", "π/4", "π/8", "π/2", "π/4", "π/2"], cs)
		stroke.(hs)
		LuxorGraphPlot.text.(Ref("H"), hs)
		fill.(dots)
		stroke.(inputs)
		#stroke.(outputs)
		stroke.(cons)
	end
end

# ╔═╡ c82d42d1-b542-464a-86ce-b294ba1794fd
md"""
The output indices are left open at this moment.
"""

# ╔═╡ 3190b0a1-3c67-412f-8002-f3d9c8cf64a3
md"""
## Compute the expectation value
"""

# ╔═╡ 1131347f-389a-4941-ad87-07480a8f97c5
observable = chain(4, [put(4, i=>X) for i in 1:4]);

# ╔═╡ 51c319c4-d729-4774-80b3-b00b5945fced
reg = zero_state(4)

# ╔═╡ 5aa12b6b-7b98-44e1-9fe5-4c34f7950aa1
expect(observable, reg=>qft)

# ╔═╡ 607ba3e1-a95d-4c05-825e-37ea538df099
extended_circuit = chain(qft, observable, qft'); vizcircuit(extended_circuit)

# ╔═╡ ccffa22f-b5e1-4c36-b93b-cba41bbbca41
# initial state
input_states = Dict([i=>zero_state(1) for i in 1:4])

# ╔═╡ 507e2118-ba05-4e7a-9b3b-54d4818401c7
qft_net = yao2einsum(extended_circuit; initial_state = input_states, final_state = input_states, optimizer = TreeSA(nslices=2))

# ╔═╡ 72c31b7e-1bc0-4518-9a67-be6a0593a431
show_einsum(qft_net.code; layout=SpringLayout(optimal_distance=25))

# ╔═╡ cd81e14e-7f1c-416b-8dd2-3ab1546995aa
contraction_complexity(qft_net)

# ╔═╡ bdef27a3-0c0f-4349-a134-0020e0e1efd0
contract(qft_net)

# ╔═╡ 5e11bfc0-d1c9-4a86-9eef-9c20488c8f93
md"# Combinatorial optimization"

# ╔═╡ ef3c35e8-c7ec-4a18-894c-7e511b598aa4
md"""
## Tensor network for solving combinatorial optimization problems: GenericTensorNetworks.jl
"""

# ╔═╡ 889a4baf-842c-4429-b472-394eb35d8600
graph = random_diagonal_coupled_graph(7, 7, 0.8)

# ╔═╡ 2fca90fd-16b4-4621-85c7-96b1ba7f82b8
show_graph(graph, StressLayout(optimal_distance=20))

# ╔═╡ d2ef3090-4786-4391-8b8a-c7d9a7f1b511
md"## Independent set problem"

# ╔═╡ 1d579198-b6e2-4b60-8759-85d3fc6ee244
md"""
TODO: from independent set problem to energy based model and partition function to tensor networks.
"""

# ╔═╡ 8ebfc7a9-973e-421b-88fe-0adc3fbd076f
problem = IndependentSet(graph)  # Independent set problem

# ╔═╡ f01b2afd-89bf-4775-a80b-1e336c7fcc60
generic_tn = GenericTensorNetwork(problem)  # Convert to tensor network

# ╔═╡ 6d14bc3f-026a-4e6a-9734-cb987a528af3
fieldnames(generic_tn |> typeof)

# ╔═╡ 7110cb89-3395-44c3-aa42-e255e03d4cd2
show_einsum(generic_tn.code)

# ╔═╡ 334dd61b-787a-4a2c-b507-448791df917a
res_size = solve(generic_tn, SizeMax())[]  # MIS size

# ╔═╡ a57f47ca-d280-4edf-8829-8ff36e906624
res_count = solve(generic_tn, CountingMax(2))[]  # Counting of independent sets with largest 2 sizes

# ╔═╡ 103bfdb6-7d4b-4a5f-963e-df8df3f94c6b
configs_raw = solve(generic_tn, ConfigsMax(2; tree_storage=true))[]  # The corresponding configurations

# ╔═╡ 796f0ab7-c0c6-41e1-b83c-331a2be6c302
configs = read_config(configs_raw)

# ╔═╡ 776ffe52-bbe2-4ad2-b3e8-8644d34e0b50
show_configs(problem, StressLayout(optimal_distance=20), fill(configs[2][1], 1, 1))

# ╔═╡ 47378ab1-5883-4ec8-ad60-2d54720c0f4c
show_landscape((x, y)->hamming_distance(x, y) <= 2, configs_raw; layout_method=:spring)

# ╔═╡ d9d22920-601f-4135-b529-bcc01caae23a
md"# Tensor network for probabilistic inference: TensorInference.jl"

# ╔═╡ a73e8709-95ab-4d9c-9312-a0731d598f18
md"""
## Package features
Solutions to the most common probabilistic inference tasks, including:
- Probability of evidence (PR): Calculates the total probability of the observed evidence across all possible states of the unobserved variables.
- Marginal inference (MAR): Computes the probability distribution of a subset of variables, ignoring the states of all other variables.
- Maximum a Posteriori Probability estimation (MAP): Finds the most probable state of a subset of unobserved variables given some observed evidence.
- Marginal Maximum a Posteriori (MMAP): Finds the most probable state of a subset of variables, averaging out the uncertainty over the remaining ones.
"""

# ╔═╡ 102aabd3-048c-4701-a384-27b525e80f9c
md"""
## The ASIA network

The graph below corresponds to the *ASIA network*, a simple Bayesian model
used extensively in educational settings. It was introduced by Lauritzen in
1988 [^lauritzen1988local].

```
┌───┐           ┌───┐
│ A │         ┌─┤ S ├─┐
└─┬─┘         │ └───┘ │
  │           │       │
  ▼           ▼       ▼
┌───┐       ┌───┐   ┌───┐
│ T │       │ L │   │ B │
└─┬─┘       └─┬─┘   └─┬─┘
  │   ┌───┐   │       │
  └──►│ E │◄──┘       │
      └─┬─┘           │
┌───┐   │   ┌───┐     │
│ X │◄──┴──►│ D │◄────┘
└───┘       └───┘
```

The table below explains the meanings of each random variable used in the
ASIA network model.

| **Random variable**  | **Meaning**                     |
|        :---:         | :---                            |
|        ``A``         | Recent trip to Asia             |
|        ``T``         | Patient has tuberculosis        |
|        ``S``         | Patient is a smoker             |
|        ``L``         | Patient has lung cancer         |
|        ``B``         | Patient has bronchitis          |
|        ``E``         | Patient hast ``T`` and/or ``L`` |
|        ``X``         | Chest X-Ray is positive         |
|        ``D``         | Patient has dyspnoea            |

---

We now demonstrate how to use the TensorInference.jl package for conducting a
variety of inference tasks on the Asia network.

---

Import the TensorInference package, which provides the functionality needed
for working with tensor networks and probabilistic graphical models.
"""

# ╔═╡ d2d7984e-9298-42ac-92b2-99b612c6ec7e
md"""
Load the ASIA network model from the `asia.uai` file located in the examples
directory. See [Model file format (.uai)](@ref) for a description of the
format of this file.
"""

# ╔═╡ c3181124-4c6a-478e-9886-02e085251ec3
model = read_model_file(pkgdir(TensorInference, "examples", "asia-network", "model.uai"))

# ╔═╡ 3bbb23af-4303-45e9-b1bc-76450524d103
md"Create a tensor network representation of the loaded model."

# ╔═╡ b96a17ae-c2b8-428b-a965-ecd09bd2fee1
inference_tn = TensorNetworkModel(model)

# ---

# Calculate the partition function. Since the factors in this model are
# normalized, the partition function is the same as the total probability, $1$.

# ╔═╡ 325f8bb8-6097-49fb-99e7-a1cb34f27116
probability(inference_tn) |> first

# ---

# Calculate the marginal probabilities of each random variable in the model.

# ╔═╡ 90e23bda-0665-4252-9675-81af96dfa04b
marginals(inference_tn)

# ---

# Retrieve all the variables in the model.

# ╔═╡ 00b47a54-cae5-4a41-9d1a-cad8e4264c3f
get_vars(inference_tn)

# ╔═╡ f419f822-088f-40fa-b1c5-93d0bdaa0137
md"""
Set the evidence: Assume that the "X-ray" result (variable 7) is negative.
Since setting the evidence may affect the contraction order of the tensor
network, recompute it.
"""

# ╔═╡ 3fd46c0c-8154-4876-b635-a43dae5e22ce
inference_tn2 = TensorNetworkModel(model, evidence = Dict(7 => 0))

# ╔═╡ e31fdb85-42ff-4169-867c-fafc35c19ebe
md"""
Calculate the maximum log-probability among all configurations.
"""

# ╔═╡ 5ad55ef1-52ea-4de9-943b-57c5aed16fad
maximum_logp(inference_tn2)

# ╔═╡ 86a1f5d6-3bb5-42f2-845f-4b615356b888
md"Generate 10 samples from the posterior distribution."

# ╔═╡ e77c4eae-6ff8-4906-aabc-683460537718
sample(inference_tn2, 10)

# ╔═╡ 2c6c1627-df8e-4d88-a7fe-4233558536ab
md"Retrieve both the maximum log-probability and the most probable configuration."

# ╔═╡ bcc4c2dd-41e3-44c2-8495-080063368cd4
logp, cfg = most_probable_config(inference_tn2)

# ╔═╡ 3c4eb3fb-bc6a-4f30-99ab-52c480340a24
md"""
Compute the most probable values of certain variables (e.g., 4 and 7) while
marginalizing over others. This is known as Maximum a Posteriori (MAP)
estimation.
"""

# ╔═╡ 091843c7-7968-4563-a71d-e89fad1d07c3
mmap = MMAPModel(model, evidence=Dict(7=>0), queryvars=[4,7])

# ╔═╡ 7425e5cd-2c3e-41b3-99a0-450d3eb54ff3
md"Get the most probable configurations for variables 4 and 7."

# ╔═╡ fb14fb1b-f868-4af8-97d1-1f9923651120
most_probable_config(mmap)

# ╔═╡ 4117497a-6055-41ce-92b6-dd2a0a151366
md"""
Compute the total log-probability of having lung cancer. The results suggest
that the probability is roughly half.
"""

# ╔═╡ 40d35ecd-8c2c-49be-b150-6ac7639fdee7
log_probability(mmap, [1, 0]), log_probability(mmap, [0, 0])

# ╔═╡ 6541b03c-53af-4697-8092-74f4f29106ea
md"""
[^lauritzen1988local]: Steffen L Lauritzen and David J Spiegelhalter. Local computations with probabilities on graphical structures and their application to expert systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 50(2):157–194, 1988.
"""

# ╔═╡ Cell order:
# ╟─ca6bee1f-d3d3-4620-aba9-363e5b856c69
# ╟─e9fbbd5d-6cea-4e1c-807b-6dc3939d3628
# ╟─92f32d6f-de53-4794-8451-5a3ca62ef828
# ╟─a481f737-c83a-4f55-b1b5-cbb33ddbc9f4
# ╟─c12e790f-769c-4277-b41d-81d08c0e134c
# ╠═b9970d7f-9a69-424e-89c0-bbfcfe5f12d7
# ╟─ec23abba-eaf0-4e7a-bf7e-b50ffbe9532f
# ╠═0a83fd07-6dd3-4228-b435-c7f0c1ddbc12
# ╠═e82c8140-c80e-4683-a72a-c8dec034c0a9
# ╠═8f153a7f-3977-44ce-8c9c-7540f6c84e2c
# ╠═4e849834-1f00-4283-ad7c-f7a783d77ab4
# ╠═0ddf1061-d53c-4561-943b-f4460ee58962
# ╟─c4e8f71a-3118-4455-a9a5-b427e2d19ba0
# ╟─6af3c70e-5a7e-4ab8-baf1-d666fea321cb
# ╠═3b9821b6-851d-4509-aa7c-7ac7c8d57399
# ╠═56cfb6d5-1866-4318-a578-7b30660b1f45
# ╠═cee4366b-a6f0-4a75-bcab-82b77b31b451
# ╠═15c74390-bb16-4ea6-8e6e-70f9628304c4
# ╟─008b26e8-862b-4e4a-8d76-fd671defca4f
# ╠═27c769f8-52ec-496c-bea7-cc210af9df99
# ╠═f93ed0d4-bb49-462e-88ad-3995c2a9532c
# ╠═ca5d4d5b-dfe3-4fd8-a5b2-6cbe70ef84a4
# ╠═6884f655-aac8-4f22-8de7-e2624f1e9246
# ╟─4d412abe-17b3-46bf-99fe-b133b20610ee
# ╠═3c423eb5-b64a-4256-a41c-7d675d7396c8
# ╟─6ce21263-bf2c-41b1-ac1f-5b4b29c94465
# ╠═c4eac832-3629-41bf-8dc9-5b2a6d3d914b
# ╠═3c84890c-0371-44cb-a9c7-ae76adeb0307
# ╟─98cafa86-1fa5-43b9-b479-3f0893fecbb8
# ╠═f80aa1e7-1a92-4a52-a070-d382fa36dcb3
# ╟─b4d09576-c8b3-4363-bd45-1364e4918862
# ╠═750ab594-9990-4ad1-a8c8-e66ea058909d
# ╟─9411e8b2-aa55-487f-b666-5eb1e3a821d9
# ╠═dce59830-0885-40fb-9693-b9f070f2388b
# ╟─63e793e3-6eb1-4c65-8041-99beb33dc1fa
# ╟─1484dee3-df3f-4aaa-8c32-78eea3133e8f
# ╟─3fc75abf-5f97-4534-87aa-7c5944e58d64
# ╟─90f342ba-0a56-4702-afd5-6a8c49964b19
# ╟─25f8e2f2-022e-4c65-aff3-4af8fe43a5dc
# ╟─33f37f98-53e9-45b4-919a-2b9b1481f4ce
# ╠═c3cc1bd0-68e6-430c-b306-696c51165b9c
# ╠═f32a5030-8c21-4859-82e8-8953cf8b534f
# ╠═7b87b462-4c32-40f7-ae8f-f68df3d42e45
# ╠═10242d23-2a97-43d5-874f-db283dfbd274
# ╠═f4e57c8d-57d3-4ee1-9cfe-06bfac8ac87f
# ╟─49dfaa9b-bcbc-4f60-8496-34561e57d098
# ╟─d96500ba-ade4-4666-a37b-222673960117
# ╟─5956ee36-d0b8-4231-a5ae-b18d5c02842e
# ╠═ddb5868a-7406-4f6f-971d-16394f4cea3b
# ╟─9671c446-775e-4cb3-8e95-7c32480eeb07
# ╟─a3f038a5-8f09-4c33-bf71-ed07e03037b3
# ╟─5af0aca3-acd3-4172-b922-91c5568fbb35
# ╟─d336e16b-4af8-4564-9205-3ae51e61c33e
# ╟─b7735171-7345-49da-916e-b8c9942eec84
# ╟─6173ea8a-a9e7-40e8-badb-601f8e96bd7d
# ╠═cb012a72-f5d8-41b7-9146-0cb184269801
# ╟─d2833db0-7f86-493b-b96e-8747f782f129
# ╟─29a41d88-7531-4914-866f-bc4185bde8c2
# ╟─ae35e03a-0da9-496f-b942-ff312fb2b366
# ╟─faafeff1-c964-440c-a2e8-640fd323f5ae
# ╟─a991e849-b397-40c6-8036-b9aa728f4c22
# ╟─e229dd53-bf34-4048-8464-61cc35225225
# ╟─c82d42d1-b542-464a-86ce-b294ba1794fd
# ╟─3190b0a1-3c67-412f-8002-f3d9c8cf64a3
# ╠═1131347f-389a-4941-ad87-07480a8f97c5
# ╠═51c319c4-d729-4774-80b3-b00b5945fced
# ╠═5aa12b6b-7b98-44e1-9fe5-4c34f7950aa1
# ╠═607ba3e1-a95d-4c05-825e-37ea538df099
# ╠═ccffa22f-b5e1-4c36-b93b-cba41bbbca41
# ╠═507e2118-ba05-4e7a-9b3b-54d4818401c7
# ╠═72c31b7e-1bc0-4518-9a67-be6a0593a431
# ╠═cd81e14e-7f1c-416b-8dd2-3ab1546995aa
# ╠═bdef27a3-0c0f-4349-a134-0020e0e1efd0
# ╟─5e11bfc0-d1c9-4a86-9eef-9c20488c8f93
# ╟─ef3c35e8-c7ec-4a18-894c-7e511b598aa4
# ╠═332085f3-ef12-4a7d-9714-e8f4a88d5a28
# ╠═889a4baf-842c-4429-b472-394eb35d8600
# ╠═5c3dfeeb-3dd2-4a4d-a1b7-2d292beccf73
# ╠═2fca90fd-16b4-4621-85c7-96b1ba7f82b8
# ╟─d2ef3090-4786-4391-8b8a-c7d9a7f1b511
# ╟─1d579198-b6e2-4b60-8759-85d3fc6ee244
# ╠═8ebfc7a9-973e-421b-88fe-0adc3fbd076f
# ╠═f01b2afd-89bf-4775-a80b-1e336c7fcc60
# ╠═6d14bc3f-026a-4e6a-9734-cb987a528af3
# ╠═7110cb89-3395-44c3-aa42-e255e03d4cd2
# ╠═334dd61b-787a-4a2c-b507-448791df917a
# ╠═a57f47ca-d280-4edf-8829-8ff36e906624
# ╠═103bfdb6-7d4b-4a5f-963e-df8df3f94c6b
# ╠═796f0ab7-c0c6-41e1-b83c-331a2be6c302
# ╠═776ffe52-bbe2-4ad2-b3e8-8644d34e0b50
# ╠═47378ab1-5883-4ec8-ad60-2d54720c0f4c
# ╟─d9d22920-601f-4135-b529-bcc01caae23a
# ╟─a73e8709-95ab-4d9c-9312-a0731d598f18
# ╟─102aabd3-048c-4701-a384-27b525e80f9c
# ╠═1845d970-4f90-4714-bc74-6e38b27160f4
# ╟─d2d7984e-9298-42ac-92b2-99b612c6ec7e
# ╠═c3181124-4c6a-478e-9886-02e085251ec3
# ╟─3bbb23af-4303-45e9-b1bc-76450524d103
# ╠═b96a17ae-c2b8-428b-a965-ecd09bd2fee1
# ╠═325f8bb8-6097-49fb-99e7-a1cb34f27116
# ╠═90e23bda-0665-4252-9675-81af96dfa04b
# ╠═00b47a54-cae5-4a41-9d1a-cad8e4264c3f
# ╟─f419f822-088f-40fa-b1c5-93d0bdaa0137
# ╠═3fd46c0c-8154-4876-b635-a43dae5e22ce
# ╟─e31fdb85-42ff-4169-867c-fafc35c19ebe
# ╠═5ad55ef1-52ea-4de9-943b-57c5aed16fad
# ╟─86a1f5d6-3bb5-42f2-845f-4b615356b888
# ╠═e77c4eae-6ff8-4906-aabc-683460537718
# ╟─2c6c1627-df8e-4d88-a7fe-4233558536ab
# ╠═bcc4c2dd-41e3-44c2-8495-080063368cd4
# ╟─3c4eb3fb-bc6a-4f30-99ab-52c480340a24
# ╠═091843c7-7968-4563-a71d-e89fad1d07c3
# ╟─7425e5cd-2c3e-41b3-99a0-450d3eb54ff3
# ╠═fb14fb1b-f868-4af8-97d1-1f9923651120
# ╟─4117497a-6055-41ce-92b6-dd2a0a151366
# ╠═40d35ecd-8c2c-49be-b150-6ac7639fdee7
# ╟─6541b03c-53af-4697-8092-74f4f29106ea
