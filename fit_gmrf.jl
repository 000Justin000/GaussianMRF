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

#-----------------------------------------------------------------------
# model dependent 1: compute the adjacency matrix of the graphical model
#-----------------------------------------------------------------------
function get_adjacency_matrices(G, p; interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]))
    """
    Given a graph, generate the graphical model that has every vertex mapped to
    p vertices, with p of them representing features

    Args:
       G: LightGraph Object
       p: number of features
       interaction_list: feature index pairs where there exist direct same-vertex interactions
    Return:
       A: an array of matrices for the graphical model:
          first p matrices: connections between same-channel features on
                            different vertices (normalized Laplacian)
          rest p(p+1) matrices: covariance among features on same vertices
    """
    n, L = nv(G), normalized_laplacian(G);

    A = Vector{SparseMatrixCSC}();

    # connections among corresponding features on different vertices
    # A_{i} = L ⊗ J_{ii}
    for i in 1:p
        push!(A, kron(L, sparse([i], [i], [1.0], p, p)));
    end

    # connections among different features on same vertices
    # A_{ii} = I ⊗ J_{ii}
    # A_{ij} = I ⊗ J_{ij}
    for (i,j) in interaction_list
        if (j == i)
            push!(A, kron(speye(n), sparse([i], [i], [1.0], p, p)));
        elseif (j>i)
            push!(A, kron(speye(n), sparse([i,j], [j,i], [1.0,1.0], p, p)));
        else
            error("unexpected pair")
        end
    end

    return A;
end

function getα(φ, p; interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]), ϵ=1.0e-5)
    """
    Log-Cholesky parametrization of the precision matrix

    Args:
       φ: p + q dimensional vector, q = p(p+1)/2 would indicate all pairwise interaction
       p: number of features
    """
    function upper_triangular(coeffs)
        Is = Vector{Int}();
        Js = Vector{Int}();
        Vs = Vector{eltype(φ)}();

        for (coeff,pair) in zip(coeffs,interaction_list)
            @assert pair[1] <= pair[2] "unexpected pair"
            push!(Is, pair[1]);
            push!(Js, pair[2]);
            push!(Vs, (pair[1] == pair[2]) ? softplus(coeff) : coeff);
        end

        return Tracker.collect(sparse(Is,Js,Vs, p,p));
    end

    @assert (length(φ) == p + length(interaction_list)) "number of parameters mismatch number of matrices"

    R = upper_triangular(φ[p+1:end]);
    Q = R'*R + ϵ*I;

    return vcat(exp.(φ[1:p]), Tracker.collect([Q[i,j] for (i,j) in interaction_list]));
end
#---------------------------------------------------------------------

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

function sample_synthetic(graph_size="medium", shift=0.0; synthetic_dict=Dict("p"=>1, "N"=>1, "α0"=>nothing), savedata=false)
    Random.seed!(0);

    if graph_size == "tiny"
        G = complete_graph(2);
    elseif graph_size == "small"
        G = connected_watts_strogatz(10, 6, 0.10);
    elseif graph_size == "medium"
        G = connected_watts_strogatz(100, 6, 0.05);
    elseif graph_size == "ring"
        G = connected_watts_strogatz(1000, 6, 0.00);
    elseif graph_size == "large"
        G = connected_watts_strogatz(1000, 6, 0.01);
    else
        error("unexpected size option");
    end

    p, N = synthetic_dict["p"], synthetic_dict["N"];

    # n: number of vertices in G
    # N: number of samples
    n = nv(G);

    # attribute interaction pairs on the same vertex
    interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]);
    A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

    #-------------------------------------------------------------
    # interactions encoded by α0
    #-------------------------------------------------------------
    #    1:p   parameters: similarity between neighboring vertices
    #  p+1:2p  parameters: diagonal elements in γ
    # 2p+1:end parameters: off diagonal elements
    #-------------------------------------------------------------
    if synthetic_dict["α0"] != nothing
        α0 = synthetic_dict["α0"];
    else
        # sample points from Gaussian distribution
        xx = randn(p,p);
        # compute the precision matrix of sampled points
        γ = inv(xx'*xx + 0.01*I);
        # parameters for the precision matrix for the GMRF model
        α0 = vcat(10.0.^(shift .+ (rand(p).-0.5)), [γ[i,j] for (i,j) in interaction_list]);
        # print parameters
        @printf("α0:    %s\n", array2str(α0)); flush(stdout);
    end

    # define a Gaussian distribution with certain covariance matrix
    CM0 = inv(Array(getΓ(α0; A=A)));
    CM = (CM0 + CM0')/2.0;
    g = MvNormal(CM);

    # sample the multi-variate Gaussian distribution to get the vertex attributes
    # the vertex attributes is give by a three dimensional tensor: attribute_type × vertex × sample
    Y = cat([reshape(rand(g), (p,n)) for _ in 1:N]..., dims=3);

    if savedata
        open("datasets/synthetic/" * graph_size * "_" * @sprintf("%+2.1f", shift) * ".json", "w") do f
            JSON.print(f, Dict("A"=>Matrix(adjacency_matrix(G)), "α0"=>α0, "Y"=>Y));
        end
    end

    return G, α0, Y;
end

function prepare_data(dataset)
    if ((match(r"^synthetic_([a-z]+)_([0-9.+-]+)_([0-9]+)_([0-9]+)", dataset) !== nothing) ||
        (match(r"^county_([0-9]+)_(.+)", dataset) !== nothing) ||
        (match(r"^facebook_(.+)", dataset) !== nothing) ||
        (match(r"^county_facebook_([0-9]+)_(.+)", dataset) !== nothing) ||
        (match(r"^climate_([0-9]+)_(.+)", dataset) !== nothing) ||
        (match(r"^ward_([0-9]+)_(.+)", dataset) !== nothing) ||
        (match(r"^twitch_(.+)_true_([0-9]+)", dataset) !== nothing) ||
        (match(r"^social_traits_([a-zA-Z]+)_([a-zA-Z]+)", dataset) !== nothing) ||
        (match(r"^cropsim_([a-z]+)_([0-9]+)_([0-9]+)", dataset) !== nothing))
        # read the graph topology G, labels, and features
        G, _, labels, feats = read_network(dataset);

        # n: number of vertices in G
        n = nv(G);

        # p: number of attributes
        p = length(feats[1]) + length(labels[1]);

        # attribute interaction pairs on the same vertex
        interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p])
        A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

        # the vertex attributes is give by a three dimensional tensor: attribute_type × vertex × sample
        Y = unsqueeze(hcat([vcat(feat,label) for (feat,label) in zip(feats,labels)]...), 3);
    else
        error("unexpected dataset")
    end

    λ_getα = φ -> getα(φ, p; interaction_list=interaction_list);

    return G, A, λ_getα, Y;
end

function fit_gmrf_once(G, A, λ_getα, Y, seed_val)
    Random.seed!(seed_val);

    n = nv(G);
    V = collect(1:size(A[1],1));

    #---------------------------------------------------------------------
    # model dependent 2: from flux parameters to α
    #---------------------------------------------------------------------
    ϕ = param(zeros(size(Y,1)));
    getρ() = reshape(ϕ, (size(Y,1),1));
    #---------------------------------------------------------------------
    φ = param(randn(length(A)));
    getα() = λ_getα(φ);
    #---------------------------------------------------------------------

    function Equadform(Y)
        batch_size = size(Y,3);
        ys = [vec(Y[:,:,i] .- getρ()) for i in 1:batch_size];

        return mean(quadformSC(getα(), ys_; A=A, L=V) for ys_ in ys);
    end

    function loss(Y; t=128, k=64)
        Ω = 0.5 * logdetΓ(getα(); A=A, P=V, t=t, k=k);
        Ω -= 0.5 * Equadform(Y);

        return -Ω/n;
    end

    n_step = 3000;
    n_batch = 1;
    N = size(Y,3);
    print_params() = (@printf("α:     %s\n", array2str(getα())); flush(stdout));
    dat = [(Y[:,:,sample(1:N, n_batch)],) for _ in 1:n_step];
    train!(loss, [Flux.params(φ)], dat, [ADAMW(1.0e-2, (0.9, 0.999), 2.5e-4)]; start_opts = [0], cb = print_params, cb_skip=n_step+1);

    return data(getρ()), data(getα()), mean(data(loss(Y; t=256, k=128)) for _ in 1:30);
end

function fit_gmrf(dataset)
    Random.seed!(0);
    G, A, λ_getα, Y = prepare_data(dataset);

    T = 32;

    ρρ = Vector{Any}(undef,T);
    αα = Vector{Any}(undef,T);
    LL = Vector{Any}(undef,T);

    print_lock = Threads.SpinLock()

    Threads.@threads for i in 1:T
        ρ, α, L = fit_gmrf_once(G, A, λ_getα, Y, i);
        ρρ[i], αα[i], LL[i] = ρ, α, L;

        lock(print_lock) do
            @printf("ρ:     %s;    α:     %s\n", array2str(ρ), array2str(α)); flush(stdout);
        end
    end

    ρ_opt = ρρ[argmin(LL)];
    α_opt = αα[argmin(LL)];

    @printf("ρ_opt: %s;    α_opt: %s\n", array2str(ρ_opt), array2str(α_opt)); flush(stdout);
end

function calculate_VI(G, p, α, lidx, fidx, dtr, dte)
    # attribute interaction pairs on the same vertex
    interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]);
    A = get_adjacency_matrices(G, p; interaction_list=interaction_list);
    Γ = getΓ(α; A=A);
    @assert isposdef(Γ);

    # the indices for features in fidx and vertices in V
    FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
    obsf = FIDX(fidx, vertices(G));
    obsl = FIDX(lidx, dtr);
    trgl = FIDX(lidx, dte);

    function schur_complement(M, idx)
        cmp = setdiff(1:size(M,1), idx);
        return M[idx,idx] - M[idx,cmp] * inv(M[cmp,cmp]) * M[cmp,idx];
    end

    Σ = inv(Matrix(Γ));
    Σ0 = Σ[vcat(trgl),vcat(trgl)];
    Σ1 = schur_complement(Σ[vcat(trgl,obsl),vcat(trgl,obsl)],           1:length(trgl));
    Σ2 = schur_complement(Σ[vcat(trgl,obsf),vcat(trgl,obsf)],           1:length(trgl));
    Σ3 = schur_complement(Σ[vcat(trgl,obsl,obsf),vcat(trgl,obsl,obsf)], 1:length(trgl));

    VI_L2_L1  = 1.0 - tr(Σ1) / (tr(Σ0) - sum(Σ0)/length(trgl));
    VI_L2_F   = 1.0 - tr(Σ2) / (tr(Σ0) - sum(Σ0)/length(trgl));
    VI_L2_FL1 = 1.0 - tr(Σ3) / (tr(Σ0) - sum(Σ0)/length(trgl));

    return VI_L2_L1, VI_L2_F, VI_L2_FL1;
end

function estimate_VI(G, p, α, lidx, fidx, dtr, dte; t=128, k=128, seed_val=0)
    Random.seed!(seed_val);

    # attribute interaction pairs on the same vertex
    interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p])
    A = get_adjacency_matrices(G, p; interaction_list=interaction_list);
    Γ = getΓ(α; A=A);
    @assert isposdef(Γ);

    # the indices for features in fidx and vertices in V
    FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
    obsf = FIDX(fidx, vertices(G));
    obsl = FIDX(lidx, dtr);
    trgl = FIDX(lidx, dte);

    vv = zeros(size(Γ,1)); vv[trgl] .= 1.0;

    VI_L2_L1  = 1.0 - trinv(Γ[vcat(trgl,obsf),vcat(trgl,obsf)]; P=1:length(trgl), t=t, k=k) / (trinv(Γ; P=trgl, t=t, k=k) - vv'*cg(Γ,vv)/length(trgl));
    VI_L2_F   = 1.0 - trinv(Γ[vcat(trgl,obsl),vcat(trgl,obsl)]; P=1:length(trgl), t=t, k=k) / (trinv(Γ; P=trgl, t=t, k=k) - vv'*cg(Γ,vv)/length(trgl));
    VI_L2_FL1 = 1.0 - trinv(Γ[vcat(trgl),vcat(trgl)];           P=1:length(trgl), t=t, k=k) / (trinv(Γ; P=trgl, t=t, k=k) - vv'*cg(Γ,vv)/length(trgl));

    return  VI_L2_L1, VI_L2_F, VI_L2_FL1;
end