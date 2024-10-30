#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": canvas, draw, tree, plot
#import "@preview/quill:0.3.0": *
#import "@preview/physica:0.9.3": *
#import "graph.typ": show-grid-graph, grid-graph-locations, show-graph, spring-layout, show-udg-graph, udg-graph, random-regular-graph
#import "@preview/pinit:0.1.3": *
#import "@preview/colorful-boxes:1.2.0": *

// #show raw.where(block: true): it=>{
//   par(justify:false,block(fill:rgb("#f0f0fe"),inset:1.5em,width:99%,text(it)))
// }

#set cite(style: "apa")

#let s = hkustgz-theme.register()
#let ket(x) = $|#x angle.r$

// Global information configuration
#let s = (s.methods.info)(
  self: s,
  title: [#grid(image("images/redbird.png", width:20pt), [Large scale tensor network contraction in Julia])],
  subtitle: [(CCF 中国开源大会 2024)],
  author: [Jin-Guo Liu],
  date: datetime.today(),
  institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
)

// Extract methods
#let (init, slides) = utils.methods(s)
#show: init

// Extract slide functions
#let (slide, empty-slide, title-slide, outline-slide, new-section-slide, ending-slide) = utils.slides(s)
#show: slides.with()

#outline-slide()

#let tensor(location, name, label) = {
  import draw: *
  circle(location, radius: 13pt, name: name)
  content((), text(black, label))
}

#let labelnode(loc, label) = {
  import draw: *
  content(loc, [$#label$], align: center, fill:white, frame:"rect", padding:0.12, stroke: none, name: label)
}
#let codebox(txt, width: auto) = {
  box(inset: 10pt, stroke: blue.lighten(70%), radius:4pt, fill: blue.transparentize(90%), text(14pt, txt), width: width)
}

#let demo() = {
  import draw: *
  labelnode((0, 0), "a")
  labelnode((0, 3), "b")
  labelnode((3, 0), "d")
  labelnode((3, 3), "c")
  labelnode((5, 1.5), "e")
  tensor((0, 1.5), "A", [])
  tensor((1.5, 0), "B", [])
  tensor((1.5, 1.5), "C", [])
  tensor((3, 1.5), "D", [])
  tensor((1.5, 3), "E", [])
  tensor((4, 0.75), "F", [])
  for (a, b) in (
    ("a", "A"),
    ("b", "A"),
    ("a", "B"),
    ("d", "B"),
    ("a", "C"),
    ("c", "C"),
    ("c", "D"),
    ("d", "D"),
    ("b", "E"),
    ("c", "E"),
    ("d", "F"),
    ("e", "F"),
    ) {
    line(a, b)
  }
}

= Large scale tensor contraction in Julia
== Vector, matrix and tensor
#v(100pt)
#align(top+center, grid([*Vector:*], [`v[i]` $arrow.r$ ], [
#canvas({
  import draw: *
  tensor((0, 0), "v", [$v$])
  line("v", (rel: (1.5, 0), to: "v"))
  content((1.7, 0), [$i$])
})
], [*Matrix:*], [`A[i,j]` $arrow.r$ ], [
#canvas({
  import draw: *
  tensor((0, 0), "A", [$A$])
  line("A", (rel: (1.5, 0), to: "A"))
  line("A", (rel: (-1.5, 0), to: "A"))
  content((-1.7, 0), [$i$])
  content((1.7, 0), [$j$])
})],
[*Tensor:*], [`T[i,j,k,...]` $arrow.r$ ], [
#canvas({
  import draw: *
  tensor((0, 0), "T", [$T$])
  line("T", (rel: (-0.8, -1.3), to: "T"))
  line("T", (rel: (-1.5, 0), to: "T"))
  line("T", (rel: (0.5, -1.5), to: "T"))
  content((-1.7, 0), [$i$])
  content((-1.1, -1.6), [$j$])
  content((0.5, -1.9), [$k$])
  content((1.5, 0.0), [$dots$])
})
], columns: 3, gutter: 20pt))

== Tensor network contraction is a modern BLAS
*Dot product*
#align(center, canvas({
  import draw: *
  tensor((0, 0), "A", [$v$])
  tensor((3, 0), "B", [$w$])
  labelnode((1.5, 0), "i")
  line("A", "i")
  line("B", "i")

  content((2.0, -0.2), [contract $lr((#v(50pt)#h(120pt))) = sum_(i) v_i w_i$])
}))

*Matrix-vector multiplication*
#align(center, canvas({
  import draw: *
  tensor((0, 0), "A", [$v$])
  tensor((3, 0), "B", [$A$])
  labelnode((1.5, 0), "j")
  line("A", "j")
  line("B", "j")
  line("B", (rel: (1.5, 0), to: "B"))
  content((rel: (1.7, 0), to:"B"), [$i$])

  content((3, -0.2), [contract $lr((#v(50pt)#h(160pt))) = sum_(j) A_(i j) v_j$])
}))

#align(bottom+right, [Note: BLAS = Basic Linear Algebra Subprograms])

== Tensor network contraction is a modern BLAS
*Tensor network contraction*
#align(center, canvas({
  import draw: *
  demo()
  content("A", [A])
  content("B", [B])
  content("C", [C])
  content("D", [D])
  content("E", [E])
  content("F", [F])
  content((5.7, 1.2), [contract $lr((#v(120pt)#h(170pt))) = sum_(a b c d e) A_(a b) B_(a d) C_(a c) D_(c d) E_(b c) F_(d e)$])
}))

- Linear, parallelizable, differentiable, $dots$

#align(bottom+right, box(width: 500pt, align(left, [$"Note: In this talk, " "tensor network" = &"einsum"\ = &"sum-product network"$])))

== Tensor network contraction order
#let poly() = {
  polygon(
  fill: blue.lighten(80%),
  stroke: blue,
  (20%, 0pt),
  (60%, 0pt),
  (80%, 2cm),
  (0%,  2cm),
  )
}
#align(canvas({
  import draw: *
  line((0, -2), (18, -2), mark : (end: "straight"))
  demo()
  hobby((3, 2), (4, 2), (5, 0), (2, 0), close: true, fill: blue.transparentize(70%), stroke:none)
  set-origin((7, 0))
  demo()
  hobby((1, -1), (2.5, 2), (3, 2.5), (4, 2.5), (5, 0), (2, -1), close: true, fill: blue.transparentize(70%), stroke:none)
  set-origin((7, 0))
  demo()
  hobby((-1, 3), (1, 4), (2, 4), (1, 2), (0, 1), close: true, fill: blue.transparentize(70%), stroke:none)
  hobby((1, -1), (2.5, 2), (3, 2.5), (4, 2.5), (5, 0), (2, -1), close: true, fill: blue.transparentize(70%), stroke:none)
  set-origin((7, 0))
  content((0, -1.5), [$dots$])
}), center)

- Constraction is performed in pair-wise manner.
- The pair-wise contraction order determines the complexity (time, space, read-write).


== The hardness of finding optimal contraction order
#align(center, box(inset: 10pt, stroke: blue)[*NP-complete*])

*Theorem @Markov2008*: Let $C$ be a quantum circuit (tensor network) with $T$ gates (tensors) and whose underlying circuit graph is $G_C$. Then $C$ can be simulated deterministically in time $T^(O(1)) exp[O(text("tw")(G_C))]$.

Tree width (measures how similar a graph is to a tree):
- Tree graphs and line graphs: $O(1)$
- $L times L$ grid graph: $O(L)$
- $n$-vertex 3-regular graph: $approx n/6$

== Heuristic search for optimal contraction order

#align(center, box(grid(image("images/2024-10-31-07-23-59.png", width: 100pt), [`OMEinsum.jl`], columns: 2, gutter: 20pt), inset: 10pt, stroke: blue))

Can handle $>10^4$ tensors!

- `GreedyMethod`: fast but not optimal
- `ExactTreewidth`: optimal but exponential time @Bouchitté2001
- `TreeSA`: heuristic local search, close to optimal, **slicing** supported @Kalachev2022
- `KaHyParBipartite` and `SABipartite`: min-cut based bipartition, better heuristic for extremely large tensor networks @Gray2021


Check the blog post for more details: https://arrogantgao.github.io/blogs/contractionorder/

== Example

Step 1: Prepare the input tensors and the contraction code.
#align(center, canvas({
  import draw: *
  demo()
  content("A", [A])
  content("B", [B])
  content("C", [C])
  content("D", [D])
  content("E", [E])
  content("F", [F])
}))

#codebox(width: 100%, [
```julia
julia> using OMEinsum  # import OMEinsum

julia> tensors = [randn(20, 20) for _ in 1:6]  # A, B, C, D, E, F
6-element Vector{Matrix{Float64}}:
...

julia> code = ein"ab,ad,ac,cd,bc,de->"  # ein"..." is a string literal that returns an EinCode object
ab, ad, ac, cd, bc, de -> 
```
])

== Example

Step 3: Get the size of each index

#codebox(width: 100%, [
```julia
julia> size_dict = OMEinsum.get_size_dict(getixsv(code), tensors)  # size of each index
Dict{Char, Int64} with 5 entries:
  'a' => 20
  'd' => 20
  'c' => 20
  'e' => 20
  'b' => 20

julia> contraction_complexity(code, size_dict)  # complexity of the contraction
Time complexity: 2^17.28771237954945
Space complexity: 2^0.0
Read-write complexity: 2^11.229419688230417
```
])

== Example

Step 4: Optimize the contraction order

#codebox(width: 100%, [
```julia
julia> optcode = optimize_code(code, size_dict, TreeSA())  # using TreeSA optimizer
SlicedEinsum{Char, DynamicNestedEinsum{Char}}(Char[], de, d -> 
├─ de
└─ dc, cd -> d
   ├─ ad, ac -> dc
   │  ├─ ad
   │  └─ ac, ca -> ac
   │     ├─ ac
   │     └─ bc, ab -> ca
   │        ├─ bc
   │        └─ ab
   └─ cd
)
```
])

== Example

Step 5: Calculate the result

#codebox(width: 100%, [
```julia
julia> contraction_complexity(optcode, size_dict)
Time complexity: 2^14.037890085142509
Space complexity: 2^8.643856189774725
Read-write complexity: 2^12.241089378860856

julia> optcode(tensors...)  # code is callable, `...` splats the tensors
0-dimensional Array{Float64, 0}:
-569.622669025289
```
])

== Julia ecosystem for tensor network contraction

#canvas({
  import draw: *
  for (x, y, text, name) in ((-10, -4, [OMEinsum \ GSoC 2019], "OMEinsum"), (0, -1, [YaoToEinsum\ @Luo2020], "YaoToEinsum"), (0, -4, [GenericTensorNetwork\ @Liu2023], "GenericTensorNetwork"), (0, -7, [TensorInference\ @Roa2024], "TensorInference")) {
    content((x, y), align(center, box(text, stroke:black, inset:10pt, width: 220pt)), name: name)
  }
  line("OMEinsum.east", "YaoToEinsum.west", mark: (end: "straight"))
  line("OMEinsum.east", "GenericTensorNetwork.west", mark: (end: "straight"))
  line("OMEinsum.east", "TensorInference.west", mark: (end: "straight"))
  content((8, -1), align(left, [#box([Quantum circuit simulation], width: 200pt)]))
  content((8, -4), align(left, [#box([Combinatorial optimization], width: 200pt)]))
  content((8, -7), align(left, [#box([Probabilistic inference], width: 200pt)]))
  content((-10, -6), align(left, [#box([Tensor network contraction engine], width: 200pt)]))
})

#align(bottom+right, [Note: GSoC: Google Summer of Code])

= Application 1: Quantum simulation

== Tensor network for quantum circuit simulation

With Yao quantum simulator, we can easily define quantum circuits and observables.

#grid([
#image("images/2024-10-29-17-01-04.png", width: 500pt)
], codebox([
```julia
julia> using Yao

# create a QFT circuit
julia> qft = EasyBuild.qft_circuit(4);
```
], width: 100%), columns: 2, gutter: 20pt)

#v(100pt)
#align(bottom+right, align(horizon)[#grid([Note: ], [#image("images/2024-10-29-20-52-30.png", width: 30pt)], [is a high-performance variational quantum circuit simulator for human.], columns: 3, gutter: 5pt)])

== Create an observable

#codebox([
```julia
# create an observable
julia> observable = chain(4, [put(4, i=>X) for i in 1:4])
nqubits: 4
chain
├─ put on (1)
│  └─ X
├─ put on (2)
│  └─ X
├─ put on (3)
│  └─ X
└─ put on (4)
   └─ X

# create input states |0000>
julia> input_states = Dict([i=>zero_state(1) for i in 1:4])
Dict{Int64, ArrayReg{2, ComplexF64, Matrix{ComplexF64}}} with 4 entries:
  4 => ArrayReg{2, ComplexF64, Array...}…
  2 => ArrayReg{2, ComplexF64, Array...}…
  3 => ArrayReg{2, ComplexF64, Array...}…
  1 => ArrayReg{2, ComplexF64, Array...}…
```
], width: 100%)



== Tensor network based simulation

#grid([
],
[
#codebox(width: 100%, [
```julia
# represent sandwiched circuits to represent expectation value: qft - observable - qft'
julia> extended_circuit = chain(qft, observable, qft');

# call the magic function `yao2einsum`
julia> qft_net = yao2einsum(extended_circuit;
    initial_state = input_states,
    final_state = input_states,
    optimizer = TreeSA(nslices=2)  # using TreeSA optimizer with 2 slices
)
TensorNetwork
Time complexity: 2^9.10852445677817
Space complexity: 2^2.0
Read-write complexity: 2^10.199672344836365

julia> contract(qft_net) # calculate <reg|qft' observable qft|reg>
0-dimensional Array{ComplexF64, 0}:
0.9999999999999993 + 0.0im
```
])
], columns: 2, gutter: 20pt)

@pan2022solving : Solving the sampling problem of the sycamore quantum circuits (53 qubits)


= Application 2: Probabilistic inference

#align(center, canvas({
  import draw: *
  let dx = 3
  let dy = 0.5
  for (x, y, name, txt) in ((-7, 0, "a", [Recent trip to #text(red)[A]sia]), (3.5, 0, "b", [Patient is a #text(red)[S]moker]), (-7, -2, "c", [#text(red)[T]uberculosis]), (0, -2, "d", [#text(red)[L]ung cancer]), (7, -2, "e", [#text(red)[B]ronchitis]), (-3.5, -4, "f", [#text(red)[E]ither T or L]), (-7, -6, "g", [#text(red)[X]-Ray is positive]), (0, -6, "h", [#text(red)[D]yspnoea])) {
    rect((x - dx, y - dy), (x + dx, y + dy), stroke: black, name: name, radius: 5pt)
    content((x, y), [#txt])
  }
  for (a, b) in (("a", "c"), ("b", "d"), ("b", "e"), ("c", "f"), ("d", "f"), ("f", "g"), ("f", "h"), ("e", "h")) {
    line(a, b, mark: (end: "straight"))
  }
  content((12, -3), box([*Tensors*\ p(A)\ p(S)\ p(T|A)\ p(L|S)\ p(B|S)\ p(E|T,L)\ p(X|B)\ p(D|E,X)], stroke: blue, inset: 10pt))
})) 

Marginal probability:

$p(L) = sum_(A, S, T, B, E, X, D) p(A) p(S) p(T|A) p(L|S) p(B|S) p(E|T,L) p(X|B) p(D|E,X)$

== Exact inference in probabilistic graphical models


Solutions to the most common probabilistic inference tasks, including:
- *Probability of evidence* (PR): Calculates the total probability of the observed evidence across all possible states of the unobserved variables
- *Marginal inference* (MAR): Computes the probability distribution of a subset of variables, ignoring the states of all other variables
- *Maximum a Posteriori Probability estimation* (MAP): Finds the most probable state of a subset of unobserved variables given some observed evidence
- *Marginal Maximum a Posteriori* (MMAP): Finds the most probable state of a subset of variables, averaging out the uncertainty over the remaining ones

Traditional methods: Junction Tree method, dynamic programming et al.

== Load a probabilistic graphical model
#codebox([
```julia
using TensorInference

model = read_model_file(pkgdir(TensorInference, "examples", "asia-network", "model.uai"))
UAIModel(nvars = 8, nfactors = 8)
 cards : [2, 2, 2, 2, 2, 2, 2, 2]
 factors : 
  Factor(1), size = (2,)
  Factor(1, 2), size = (2, 2)
  Factor(3), size = (2,)
  Factor(3, 4), size = (2, 2)
  Factor(3, 5), size = (2, 2)
  Factor(2, 4, 6), size = (2, 2, 2)
  Factor(6, 7), size = (2, 2)
  Factor(5, 6, 8), size = (2, 2, 2)
```
], width: 100%)

== Create a tensor network representation

#codebox([
```julia
# Create a tensor network representation of the loaded model.
julia> inference_tn = TensorNetworkModel(model, evidence = Dict(7 => 0), optimizer=TreeSA())
TensorNetworkModel{Int64, OMEinsum.SlicedEinsum{Int64, OMEinsum.DynamicNestedEinsum{Int64}}, Array{Float64}}
variables: 1, 2, 3, 4, 5, 6, 7 (evidence → 0), 8
contraction time = 2^6.0, space = 2^2.0, read-write = 2^7.066

# Calculate the partition function
julia> probability(inference_tn)
exp(-2.2046416559839406) * fill(1.0)
```
], width: 100%)

== Calculate the marginal probabilities
#codebox([
```julia
# Calculate the marginal probabilities of each random variable in the model.
julia> marginals(inference_tn)
Dict{Vector{Int64}, Vector{Float64}} with 8 entries:
  [8] => [0.640766, 0.359234]
  [3] => [0.687754, 0.312246]
  [1] => [0.0131555, 0.986844]
  [5] => [0.506326, 0.493674]
  [4] => [0.488711, 0.511289]
  [6] => [0.57604, 0.42396]
  [7] => [1.0]
  [2] => [0.0924109, 0.907589]
```
], width: 100%)

== Calculate the maximum a posteriori probability
#codebox([
```julia
julia> logp, cfg = most_probable_config(inference_tn)
(-3.65222179200233, [1, 1, 0, 0, 0, 0, 0, 0])
```
], width: 100%)

== Sample from the model
#codebox([
```julia
julia> sample(inference_tn, 10)
10-element TensorInference.Samples{Int64}:
 [1, 1, 0, 1, 1, 1, 0, 1]
 [1, 1, 0, 0, 1, 0, 0, 0]
 [1, 1, 0, 0, 0, 0, 0, 0]
 [1, 1, 0, 0, 0, 0, 0, 0]
 ⋮
 [1, 1, 0, 0, 0, 0, 0, 1]
 [1, 1, 0, 0, 1, 0, 0, 0]
 [1, 1, 0, 0, 1, 0, 0, 1]
 [1, 1, 0, 1, 1, 1, 0, 1]
```
])


= Application 3: Combinatorial optimization

#canvas({
  import draw: *
  let desc(loc, title, description) = {
    content(loc, [#text(blue, [*#title*:]) #description])
  }
  circle((0, 0), radius: (8, 4))
  circle((4, 0), radius: (4, 2), fill:aqua)
  desc((12, 4), "NP", [nondeterministic polynomial\ - decision problems that verifiable in polynomial time])
  desc((-4, 0), "P", [polynomial time \ solvable])
  desc((4, 0.8), "NP-complete", [\ hardest in NP])
  circle((-4, 0), radius: (3, 3))
  for (i, j, name) in ((0, 2, "A"), (0.6, 0, "B"), (-2, -1, "C"), (3, -1.5, "D"), (4, -0.5, "E")) {
    circle((i, j), radius:0.2, fill: black, name:name)
  }
  for (a, b) in (("A", "B"), ("C", "B")) {
    line(a, b, mark:(end:"straight"), stroke:(thickness:2pt, paint:black))
  }
  for (a, b) in (("D", "B"), ("E", "B"), ("D", "E")) {
    line(a, b, mark:(end:"straight", start: "straight"), stroke:(thickness:2pt, paint:black))
  }
  line((11, 1), (13, 1), mark:(end:"straight"), stroke:(thickness:2pt, paint:black))
  content((14, -1), block([A is _reducible_ to B: Map problem A to problem B, the solution of B can be mapped to a solution of A.], width:250pt))
})

== NP-complete problems

=== Yes instances

- Ground state of a spin-glass Hamiltonian.
- Closest lattice vector problem.
- Is a theorem provable in $k$ steps?
- #text(red)[The maximum Independent Set problem (MIS).]

=== No instances
- Given a large integer $z$, find its prime factors. (not hard enough)
- The number of 3-coloring of a graph. (not a decision problem)

#place(top + right, figure(image("images/nature.jpg", width:17%), caption: text(14pt)[*"The nature of computation"*\ By: Cristopher Moore and Stephan Mertens], supplement: none), dy: -0pt)

#place(top, dx:350pt, dy: 50pt, canvas({
  import draw: *
  line((0, 0), (2.25, 2), stroke:black, mark:(end:"o"))
  show-grid-graph(4, 4, unitdisk: 1.1, gridsize: 1, radius: 0.2, a:(1, 0), b:(0.3, 0.9))
  content((2.2, 2.3), [$?$])
}))

== Combinatorial optimization

#grid([#image("images/configs.png", width: 80%)
@Liu2023
],
[
  #codebox([
```julia
julia> using GenericTensorNetworks, Graphs

julia> graph = smallgraph(:petersen);

julia> problem = IndependentSet(graph);

julia> net = GenericTensorNetwork(problem; optimizer=TreeSA());

julia> solve(net, CountingMax(2))  # count maximum two sizes
0-dimensional Array{Max2Poly{Float64, Float64}, 0}:
30.0*x^3 + 5.0*x^4

julia> solve(net, ConfigsMax())  # enumerating MISs
0-dimensional Array{CountingTropical{Float64, ConfigEnumerator{10, 1, 1}}, 0}:
(4.0, {1010000011, 1001001100, 0101010001, 0100100110, 0010111000})ₜ
```
])\
#text(14pt)[Source code available on GitHub: #link("https://github.com/QuEraComputing/GenericTensorNetworks.jl")[GenericTensorNetworks.jl]]
], columns: 2)

== Property extraction
#align(center, canvas({
  import draw: *
  circle((6, -5.5), radius: (10, 4), stroke: none, fill: green.transparentize(80%))
  content((6, -10), [contraction])
  for (x, y, text) in ((-9, -2, "hard problem"), (-6, -4, "energy model"), (-3, -2, "partition function"), (0, -4, "tensor network"), (6, -4, "contraction order"), (12, -4, "algebra"), (6, -6, "contract"), (6, -8, "Properties")) {
    content((x, y), box(text, stroke:black, inset:7pt), name: text)
  }
  for (a, b) in (("hard problem", "energy model"), ("energy model", "partition function"), ("partition function", "tensor network"), ("tensor network", "contract"), ("contraction order", "contract"), ("tensor network", "contraction order"), ("algebra", "contract"), ("contract", "Properties")) {
    line(a, b, mark: (end: "straight"))
  }
  content((12.3, -1.5), [- Real
  - Tropical
  - Set
  - ...])
  content((6, -1.5), [- Bipartition
  - Local search
  - PMC
  - ...])
}))


// == The efficient sampling of low energy space

// #align(center, canvas({
//   import draw: *
//   show-grid-graph(8, 8, filling:0.8, unitdisk: 1.5, gridsize: 1.2, radius: 0.2)
//   content((4, -1), [King's subgraph (Physical)])

//   // 3-regular graph
//   set-origin((16, 4))
//   let n = 50
//   let edges = random-regular-graph(n, 3)
//   let locs = spring-layout(n, edges, optimal_distance:0.8)
//   show-graph(locs, edges)
//   content((2, -5), [3-regular graph])
// }))

== A unified framework for problem reductions

#link("https://github.com/GiggleLiu/ProblemReductions.jl")[ProblemReductions.jl] (#box(image("images/ospp.jpeg"), height:30pt, baseline: 30%) Open source promotion plan)

#place(top + right, [#align(center, [#image("images/xiaofeng.png", width:50pt) Xiaofeng Li])])

#align(center, canvas({
  import draw: *
  for (x, y, text) in (
      (0, 0, "Independent set"),
      (0, 2, "QUBO (or Spin glass)"),
      (0, 6, "Dominating set"),
      (0, 4, "Max cut"),
      (-8, 1, "Vertex coloring"),
      (-8, 3, "k-SAT"),
      (-8, 5, "Circuit SAT"),
      (-14, 0, "Matching"),
      (8, 0, "Independent set on UDG"),
      (8, 2, "QUBO on grid"),
      (-14, 2, "Factoring"),
      (-14, 6, "Vertex covering"),
      (-8, -1, "Set packing"),
      (-14, 4, "Set covering")
    ){
    content((x, y), box(text, stroke:black, inset:7pt), name: text)
  }
  let arr = "straight"
  for (a, b, markstart, markend, color) in (
    ("Set covering", "Vertex covering", none, arr, gray),
    ("k-SAT", "Vertex coloring", none, arr, black),
    ("QUBO (or Spin glass)", "Max cut", arr, arr, black),
    ("k-SAT", "Independent set", none, arr, black),
    ("Independent set on UDG", "Independent set", arr, arr, black),
    ("Factoring", "Circuit SAT", none, arr, black),
    ("QUBO (or Spin glass)", "Circuit SAT", arr, none, black),
    ("Vertex covering", "Set covering", none, arr, black),
    ("Dominating set", "k-SAT", arr, none, black),
    ("Set packing", "Independent set", arr, arr, black),
    ("k-SAT", "Independent set", none, arr, black),
    ("Factoring", "Independent set on UDG", none, arr, black),
    ("k-SAT", "Circuit SAT", arr, none, black),
    ("Independent set on UDG", "QUBO on grid", arr, none, black),

    ("QUBO (or Spin glass)", "QUBO on grid", arr, arr, gray),
    ("Matching", "Circuit SAT", arr, none, gray),
    ("Vertex covering", "Circuit SAT", arr, none, gray),
  ){
    line(a, b, mark: (end: markend, start: markstart), stroke: color)
  }
  rect((4, -1), (12, 3), stroke:(dash: "dashed"))
  content((8, 3.5), [Two dimensional])
}))

== Thank you!

#align(center, grid(box(width: 200pt)[#image("images/OMEinsum.png", width:150pt)
OMEinsum.jl],
box(width: 200pt)[#image("images/GenericTensorNetworks.png", width:150pt)
GenericTensorNetworks.jl
],
box(width: 200pt)[#image("images/TensorInference.png", width:150pt)
TensorInference.jl
],
box(width: 200pt)[#image("images/Yao.png", width:150pt)
Yao.jl
],
columns: 4, gutter: 10pt))

#align(bottom+right, text(fill:blue, "jinguoliu@hkust-gz.edu.cn"))

==

#bibliography("refs.bib")