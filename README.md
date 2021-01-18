## GaussianMRF

### This repository hosts the code and some example data for the following paper:  
[An Attributed Data Model for Graph Learning Algorithms](https://arxiv.org/abs/2002.XXXXX)  
[Junteng Jia](https://000justin000.github.io/), and [Austin R. Benson](https://www.cs.cornell.edu/~arb/)  
arXiv:2002.XXXXX, 2021.

Our paper propose a generative model that unifies traditional label propagation algorithm and graph neural networks. Our model assume the node features and labels are jointly sampled from a Gaussian Markov random field (MRF). Therefore we can derive discriminative learning algorithms by computing the expectation of outcomes of the unknown nodes conditioned on different observations. In particular, the expectation of the unkonwn outcomes
- conditioned on the observed outcomes ⟶ **Label Propagation**.
- conditioned on all the features ⟶ **Linear Graph Convolution**.
- conditioned on the observed outcomes and the features ⟶ **Linear Graph Convolution + Residual Propagation**.

Our code is tested under in Julia 1.4.1, you can install all dependent packages by running.
```
julia env.jl
```

### Usage
Our code implement the three algorithms outlined above, as well as some baselines and variants. In order to use our code for your own prediction problem, the following information is required:
- **G:** the topology of your graph as a LightGraph object 
- **feats:** an array of feature vectors
- **labels:** an array of real-valued outcomes

All prediction algorithms considered in our paper are implement as one single function [run_dataset](predict.jl#L18) with different options. Each algorithm can be decomposed into three components: 1) feature pre-processing, 2) inductive prediction, 3) residual propagation. 

| **Algorithm** <td colspan=3>**step 1 options** <td colspan=3>**step 2 options** <td colspan=3>**step 3 options**|
|-
|**Options** <td colspan=1>feature_smoothing <td colspan=1>feature_smoothing |   |  
|LP            |   |   |   |
|LR       |   |   |   |
|LGC      |   |   |   |
|SGC      |   |   |   |
|GCN      |   |   |   |
|LGC/RP   |   |   |   |
|SGC/RP   |   |   |   |
|GCN/RP   |   |   |   |


LP-GNN algorithm requires minimal implementation overhead on top of standard GNN. The following is code snippet from [example_lpgnn.jl](examples/example_lpgnn.jl#L18) that predicts county-level election outcomes with demographical features.
```julia
#---------------------------------------------------------------------------------------------
# read the four requirements as listed above
#---------------------------------------------------------------------------------------------
G, A, labels, feats = read_network(network_trans);
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# define and train GNN
#---------------------------------------------------------------------------------------------
# encoder that embed vertices to vector representations
enc = graph_encoder(length(feats[1]), dim_out, dim_h, repeat(["SAGE_Mean"], 2); Ïƒ=relu);
# regression layer that maps representation to prediction
reg = Dense(dim_out, 1); 
# GNN prediction 
getRegression = L -> vcat(reg.(enc(G, L, u->feats[u]))...);
# training
Flux.train!(L->mse(labels[L], getRegression(L)), params(enc, reg), mini_batches, ADAM(0.001));
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# LP-GNN testing
#---------------------------------------------------------------------------------------------
# Î“: normalized Laplacian matrix
# L: training vertices
# U: testing vertices
# rL: GNN predicted residual for testing vertices
# lU: LP-GNN predicted outcomes for testing vertices
#---------------------------------------------------------------------------------------------
pL = getRegression(L)
pU = getRegression(U)

rL = labels[L] - data(pL);
lU = pU + cg(Î“[U,U], -Î“[U,L]*rL);
#---------------------------------------------------------------------------------------------
```
In the algorithm above, only the last 4 lines differ from the standard GNN.

The C-GNN algorithm is slightly more involving since it need to optimize the framework parameters to fit the observed correlation pattern. An example is given in [example_cgnn.jl](examples/example_cgnn.jl), which only introduce tens of lines additional code comparing to the standard GNN algorithm.

In order to run the examples, you can simply use:
```julia
julia examples/example_cgnn.jl
julia examples/example_lpgnn.jl
```


### Reproduce Experiments in Paper
The experiments in our paper can be reproduced by running.
```
bash run.sh
```
which would write the outputs to [/logs](/logs).

If you have any questions, please email to [jj585@cornell.edu](mailto:jj585@cornell.edu).
