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

### Basic Usage
Our code implement the three algorithms outlined above, as well as some baselines and variants. In order to use our code for your own prediction problem, the following information is required:
- **G:** the topology of your graph as a LightGraph object 
- **feats:** an array of feature vectors
- **labels:** an array of real-valued outcomes

All prediction algorithms considered in our paper are implement as one single function [run_dataset](predict.jl#L100) with different options (e.g. the smoothing level α, nonlinear activation σ, number of GCN layers K). Each algorithm can be decomposed into three steps: 1) feature pre-processing, 2) inductive prediction, 3) residual propagation. The options corresponding to the algorithms consider in our paper can be summarized as follows:

| <td colspan=2 align=center>**step 1** <td colspan=3 align=center>**step 2** <td colspan=2 align=center>**step 3**
|-
|**Options** <td>**feature_smoothing** <td>**α** <td>**predictor** <td>**σ**    <td>**K**     <td>**residual_propagation** <td>**α**
|LP          <td>false                 <td>      <td>mean          <td>         <td>          <td>true                     <td>[0,1]
|LR          <td>false                 <td>      <td>mlp           <td>identity <td>          <td>false                    <td>
|LGC         <td>true                  <td>[0,1] <td>mlp           <td>identity <td>          <td>false                    <td>
|SGC         <td>false                 <td>      <td>gcn           <td>identity <td>{1,2,...} <td>false                    <td>
|GCN         <td>false                 <td>      <td>gcn           <td>relu     <td>{1,2,...} <td>false                    <td>
|LGC/RP      <td>true                  <td>[0,1] <td>mlp           <td>identity <td>          <td>true                     <td>[0,1]
|SGC/RP      <td>false                 <td>      <td>gcn           <td>identity <td>{1,2,...} <td>true                     <td>[0,1]
|GCN/RP      <td>false                 <td>      <td>gcn           <td>relu     <td>{1,2,...} <td>true                     <td>[0,1]


For example, to run LGC/RP on the 2016 U.S. dataset to predict election outcomes, you can simply do the following:
```julia
# read the four requirements as listed above
G, _, labels, feats = read_network("county_facebook_2016_election");

# random split, 30% training and 70% testing
ll, uu = rand_split(nv(G), [0.3, 0.7]);

# prediction routine
run_dataset(G, feats, labels, ll, uu; 
            feature_smoothing=true, α=0.80, predictor="mlp", σ=identity, residual_propagation=true)
```
Please see [example.jl](examples/example.jl) for more details.


### Fitting Gaussian MRF
In order to fit a dataset to our proposed Gaussian MRF, one can simply run the following:
```julia
# fit dataset to a Gaussian MRF
fit_gmrf("county_facebook_2016_election")
```
This function would call [read_network](read_network.jl#L320) internally to collect the graph topology and node attributes. It will print out all the Gaussian MRF parameters after finish.


### Using Your Own Data
To use our code for your data, you simply need to add your data to the loader [read_network](read_network.jl#L320). One thing you probably want to do is to normalize each node feature (as well as the outcome) to have zero mean. For a simple example of how to write the data loader, see [read_ward](read_network.jl#L187).


### Reproduce Experiments in Paper
The experiments in our paper can be reproduced by running.
```
bash run.sh
```
which would write the outputs to [/results](/results). This potentially takes a long time due to hyperparameter scanning.

If you have any questions, please email to [jj585@cornell.edu](mailto:jj585@cornell.edu).
