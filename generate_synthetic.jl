using Flux;
using Flux.Optimise;
using Flux: train!, throttle, Tracker, unsqueeze;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using JSON;
using Plots;
using Random;
using Distributions;
using GraphSAGE;

include("utils.jl");
include("kernels.jl");
include("read_network.jl");

function connected_watts_strogatz(n, k, p; num_trials=1000)
    G = watts_strogatz(n, k, p);
    ct_trials = 1;
    while !is_connected(G) && (ct_trials < num_trials)
        G = watts_strogatz(n, k, p);
        ct_trials += 1;
    end

    @assert is_connected(G) "max number of trails reached, graph is still not connected"

    return G;
end

function sample_synthetic(graph_type="WattsStrogatz", shift=0.0; synthetic_dict=Dict("p"=>1, "N"=>1, "ξ0"=>nothing), savedata=false)
    Random.seed!(0);

    if graph_type == "Tiny"
        G = complete_graph(2);
    elseif graph_type == "Ring"
        G = connected_watts_strogatz(1000, 6, 0.00);
    elseif graph_type == "WattsStrogatzSmall"
        G = connected_watts_strogatz(100, 6, 0.05);
    elseif graph_type == "WattsStrogatz"
        G = connected_watts_strogatz(1000, 6, 0.01);
    elseif graph_type == "StochasticBlockModelSmall"
        G = stochastic_block_model(5, 1, [20, 20, 20, 20, 20]);
    else
        error("unexpected option");
    end

    p, N = synthetic_dict["p"], synthetic_dict["N"];

    # n: number of vertices in G
    # N: number of samples
    n = nv(G);

    # attribute interaction pairs on the same vertex
    interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]);
    A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

    #-------------------------------------------------------------
    # interactions encoded by ξ0
    #-------------------------------------------------------------
    #    1:p   parameters: similarity between neighboring vertices
    #  p+1:2p  parameters: diagonal elements in γ
    # 2p+1:end parameters: off diagonal elements
    #-------------------------------------------------------------
    if synthetic_dict["ξ0"] != nothing
        ξ0 = synthetic_dict["ξ0"];
    else
        # sample points from Gaussian distribution
        xx = randn(p,p);
        # compute the precision matrix of sampled points
        γ = inv(xx'*xx + 0.01*I);
        # parameters for the precision matrix for the GMRF model
        ξ0 = vcat(10.0.^(shift .+ (rand(p).-0.5)), [γ[i,j] for (i,j) in interaction_list]);
        # print parameters
        @printf("ξ0:    %s\n", array2str(ξ0)); flush(stdout);
    end

    # define a Gaussian distribution with certain covariance matrix
    CM0 = inv(Array(getΓ(ξ0; A=A)));
    CM = (CM0 + CM0')/2.0;
    g = MvNormal(CM);

    # sample the multi-variate Gaussian distribution to get the vertex attributes
    # the vertex attributes is give by a three dimensional tensor: attribute_type × vertex × sample
    Y = cat([reshape(rand(g), (p,n)) for _ in 1:N]..., dims=3);

    if savedata
        open("datasets/synthetic/" * graph_type * "/shift_" * @sprintf("%+2.1f", shift) * ".json", "w") do f
            JSON.print(f, Dict("A"=>Matrix(adjacency_matrix(G)), "ξ0"=>ξ0, "Y"=>Y));
        end
    end

    return G, ξ0, Y;
end
