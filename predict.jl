using Flux;
using Flux.Optimise;
using Flux: train!, throttle, Tracker, unsqueeze;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots;
using Distributions;
using Random;
using GraphSAGE;

include("utils.jl");
include("kernels.jl");
include("read_network.jl");

function explicit_smoothing(FT, N; α=0.5, K=256)
    """
    Args:
        FT: d×n dimensional feature matrix
         N: n×n dimensional normalized Laplacian matrix
         α: mixing parameter
         K: number of label propagation steps

    Returns:
        FS: smoothed feature matrix, FS(K) = (1-α) ∑_{i=0}^{K-1} FT (α S)^{i} + FT (α S)^{K}
    """
    S = I-N;

    FS = zeros(size(FT));
    for i in 0:K-1
        # at this point, FT == FT0 (α S)^{i}
        FS += (1-α) * FT;
        FT *= (α*S);
        # at this point, FT == FT0 (α S)^{i+1}
    end
    FS += FT;

    return FS;
end

function interpolate(L, rL; Γ)
    """
    Args:
         L: mini_batch indices for estimating noise
        rL: noise over the mini_batch L
         Γ: label propagation matrix

    Returns:
         r: noise over all vertices
    """
    n = size(Γ,1);
    U = setdiff(1:n, L);
    rU = vcat([cg(Γ[U,U], -Γ[U,L]*rL[i,:])' for i in 1:size(rL,1)]...);

    r = zeros(size(rL,1), n);
    r[:,L] = rL;
    r[:,U] = rU;

    return r;
end

function estimate_residual(U, L; LBL, pL, Γ)
    """
    Args:
          U: vertices to predict
          L: vertices with ground truth labels
        LBL: ground truth labels on L
         pL: predicted labels
          Γ: label propagation matrix

    Returns:
         lU: predictive label = base predictor output + estimated noise
    """

    # for regression task, residual is defined as ``true-label minus predicted-label''
    rL = LBL - pL;
    rU = interpolate(L, rL; Γ=Γ)[:,U];

    return rU;
end

# learn and test accuracy
function run_dataset(G, feats, labels, ll, uu; feature_smoothing=false, predictor="mean", residual_propagation=false, α=0.5, K=2, σ=relu, n_step=500, cb_skip=100, seed_val=0, return_predictions=false, return_trace=false, return_predmap=false)
    """
    Args:
          G: an undirected graph instance from LightGraphs.jl
      feats: an array of array, each provides the features on one node
     labels: an array of real-valued outcomes on all nodes
         ll: training vertices
         uu: testing vertices

    Returns:
         lU: predictive label = base predictor output + estimated noise
    """

    # fix seed_val for reproducability
    Random.seed!(seed_val);

    # FT: nf×n dimensional feature matrix
    # LB: nc×n dimensional label matrix, for regression task we consider herein, we have nc=1
    FT = hcat(feats...);
    LB = hcat(labels...);

    # the mini_batch size is 10% of the vertices
    n_batch = Int(ceil(length(ll)*0.1));
    
    # for regression task, we use R2 as the metric
    # for binary classification task, we directly output the prediction scores
    metric = return_predictions ? (x,y) -> (x,y) : R2;

    # normalized adjacency matrix
    X = normalized_laplacian(G);

    # perform feature smoothing with fixed mixing parameter α
    FF = feature_smoothing ? explicit_smoothing(FT, X; α=α) : FT;

    # number of features, number of classes for labels
    dim_f = size(FF,1);
    dim_c = size(LB,1);
    dim_h = 32;

    # predmap is the learned (inductive) predictor that takes as input: 
    # 1) the graph G, 
    # 2) the (preprocessed) feature
    # 3) the indices for nodes to be predicted
    # and output the predicted outcomes on all vertices
    if predictor == "mean"
        predmap = (G, FF, U) -> ones(dim_c,length(U)) .* mean(LB[:,ll],dims=2);
        θ = Flux.params();
        optθ = nothing;
    elseif predictor == "mlp"
        # 2-layer MLP, if σ is the identity function, this is equivalent to linear regression
        enc = Chain(Dense(dim_f, dim_h, σ), Dense(dim_h, dim_h, σ));
        cls = Chain(Dense(dim_h, dim_c));
        out = Chain(identity);
        predmap = (G, FF, U) -> out(cls(enc(FF[:,U])));
        θ = Flux.params(enc, cls, out);
        optθ = ADAMW(1.0e-3, (0.9,0.999), 2.5e-4);
    elseif predictor == "gcn"
        # K-layer GCN, if σ is the identity function, this is equivalent to SGC
        enc = graph_encoder(dim_f, dim_h, dim_h, repeat(["SAGE_GCN"], K); ks=repeat([30], K), σ=σ);
        cls = Chain(Dense(dim_h, dim_c));
        out = Chain(identity);
        predmap = (G, FF, U) -> out(cls(hcat(enc(G, U, u->FF[:,u])...)));
        θ = Flux.params(enc, cls, out);
        optθ = ADAMW(1.0e-3, (0.9,0.999), 2.5e-4);
    elseif typeof(predictor) != String
        # for inductive learning: in this case, the predictor is already a predmap function, which is learned on a different graph
        predmap = predictor
        θ = Flux.params();
        optθ = nothing;
    else
        error("unexpected predictor type");
    end

    predict = U -> predmap(G, FF, U);

    # loss function, compute batch by batch
    # this is to prevent potential memory overflow
    function batch_loss(L; bsize=500)
        loss = L -> Flux.mse(predict(L), LB[:,L]);

        nL = length(L);
        LS = [L[i:min(i+bsize-1,nL)] for i in 1:bsize:nL];

        Ω = mean(loss(L_) for L_ in LS);

        return Ω;
    end

    # label/residual propagation matrix
    # if Γ is the identity matrix, then it is ``turned off''
    Γ = residual_propagation ? I + (α/(1-α))*X : speye(nv(G));

    # the inductive predction, plus the residual propagation corrections
    function predmap_with_rp(G, FF, U, L, LBL, Γ)
        function data_batch_predict(U; bsize=500)
            nU = length(U);
            Us = [U[i:min(i+bsize-1,nU)] for i in 1:bsize:nU];
            pUs = [data(predmap(G, FF, U_)) for U_ in Us];
            pU = hcat(pUs...);

            return pU;
        end

        pL = data_batch_predict(L);
        pU = data_batch_predict(U);
        rU = estimate_residual(U,L; LBL=LB[:,L], pL=pL, Γ=Γ);

        return pU+rU;
    end

    function evaluation()
        return metric(predmap_with_rp(G, FF, uu, ll, LB[:,ll], Γ), LB[:,uu]);
    end

    #----------------------
    evaluations = [];
    #----------------------
    # this is the call back function for the training routine
    # this function is called every cb_skip steps
    #----------------------
    function cb()
        push!(evaluations, evaluation());
    end
    #----------------------

    #----------------------
    if (typeof(predictor) == String) && (predictor != "mean")
        #-------------------
        mini_batches = [tuple(sample(ll, n_batch, replace=false)) for i in 1:n_step];
        #-------------------
        train!(batch_loss, [θ], mini_batches, [optθ]; cb=cb, cb_skip=cb_skip);
        #-------------------
    else
        for _ in cb_skip:cb_skip:n_step
            cb();
        end
    end
    #----------------------

    outputs = evaluation();
    return_trace && (outputs = tuple(outputs..., evaluations));
    return_predmap && (outputs = tuple(outputs..., predmap))

    return outputs;
end

function run_transductive(G, labels, feats; compute_VI=false, ξ=nothing, lidx=nothing, fidx=nothing)
    """
    Args:
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    p = length(feats[1]) + length(labels[1]);
    #-----------------------------------------------------------------------------------
    n_step = 500;
    cb_skip = 100;
    cv_fold = 5;
    #-----------------------------------------------------------------------------------
    # ntrials = 5
    ntrials = 10;
    #-----------------------------------------------------------------------------------
    ss = collect(cb_skip:cb_skip:n_step)
    Ks = [1, 2, 3];
    # αs = [0.00, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99];
    αs = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 0.999];
    #-----------------------------------------------------------------------------------

    #--------------------------------
    for split_ratio in 0.30:0.10:0.30
    # for split_ratio in 0.10:0.10:0.60
    #--------------------------------
        if compute_VI
        vi_ll    = zeros(ntrials);
        vi_ff    = zeros(ntrials);
        vi_fl    = zeros(ntrials);
        end
        #-------------------------
        ac_lr    = zeros(ntrials);
        #-----------------------------
        ac_sgc   = zeros(ntrials);
        ac_gcn   = zeros(ntrials);
        #-------------------------
        ac_lp    = zeros(ntrials);
        ac_lgc   = zeros(ntrials);
        ac_lgcrp = zeros(ntrials);
        ac_sgcrp = zeros(ntrials);
        ac_gcnrp = zeros(ntrials);
        #-------------------------
        Kx_sgc   = zeros(Int, ntrials);
        Kx_gcn   = zeros(Int, ntrials);
        #-------------------------
        αx_lp    = zeros(ntrials);
        αx_lgc   = zeros(ntrials);
        αx_lgcrp = zeros(ntrials);
        αx_sgcrp = zeros(ntrials);
        αx_gcnrp = zeros(ntrials);
        #-------------------------

        #-----------------------------
        for seed_val in 1:ntrials
            #-------------------------
            dtr, dte = rand_split(nv(G), [split_ratio, 1.0-split_ratio]; seed_val=seed_val);
            #-------------------------

            #-------------------------
            if compute_VI
            vi_ll[seed_val], vi_ff[seed_val], vi_fl[seed_val] = calculate_VI(G, p, ξ, lidx, fidx, dtr, dte);
            end
            #-------------------------

            #-------------------------
            # split training data into a number of folds
            #-------------------------
            dVAs = rand_split(dtr, ones(cv_fold)/cv_fold; seed_val=seed_val);
            #-------------------------

            #-------------------------
            va_lr      = zeros(length(ss));
            #-------------------------
            va_sgc     = zeros(length(ss), length(Ks));
            va_gcn     = zeros(length(ss), length(Ks));
            #-------------------------
            for dVA in dVAs
                #-------------------------
                dTR = setdiff(dtr, dVA);
                #-------------------------

                #-------------------------
                # cross validation for LR
                #-------------------------
                va_lr[:]    += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="mlp", residual_propagation=false,      σ=identity, n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
                #-------------------------

                #-------------------------
                # cross validation for SGC, GCN
                #-------------------------
                for (i,K) in enumerate(Ks)
                va_sgc[:,i] += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn", residual_propagation=false, K=K, σ=identity, n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
                va_gcn[:,i] += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn", residual_propagation=false, K=K, σ=relu,     n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
                end
                #-------------------------
            end
            #-------------------------
            sx_lr  = ss[argmax(va_lr)];
            #-------------------------
            sx_sgc = ss[argmax(va_sgc)[1]];
            sx_gcn = ss[argmax(va_gcn )[1]];
            #-------------------------
            Kx_sgc[seed_val] = Ks[argmax(va_sgc)[2]];
            Kx_gcn[seed_val] = Ks[argmax(va_gcn)[2]];
            #-------------------------

            #-------------------------
            va_lp    = zeros(length(αs));
            va_lgc   = zeros(length(ss), length(αs));
            va_lgcrp = zeros(length(ss), length(αs));
            va_sgcrp = zeros(length(αs));
            va_gcnrp = zeros(length(αs));
            #-------------------------
            for dVA in dVAs
                #-------------------------
                dTR = setdiff(dtr, dVA);
                #-------------------------

                #-------------------------
                # cross validation for LP, LGC, LGC/RP, SGC/RP, GCN/RP
                #-------------------------
                for (i,α) in enumerate(αs)
                va_lp[i]      += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="mean", residual_propagation=true,  α=α,                                                                   seed_val=seed_val, return_trace=false);
                va_lgc[:,i]   += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=true,  predictor="mlp",  residual_propagation=false, α=α,                      σ=identity, n_step=n_step,  cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
                va_lgcrp[:,i] += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=true,  predictor="mlp",  residual_propagation=true,  α=α,                      σ=identity, n_step=n_step,  cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
                va_sgcrp[i]   += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=α, K=Kx_sgc[seed_val],  σ=identity, n_step=sx_sgc,  cb_skip=cb_skip, seed_val=seed_val, return_trace=false);
                va_gcnrp[i]   += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=α, K=Kx_gcn[seed_val],  σ=relu,     n_step=sx_gcn,  cb_skip=cb_skip, seed_val=seed_val, return_trace=false);
                end
                #-------------------------
            end
            #-------------------------
            sx_lgc   = ss[argmax(va_lgc  )[1]];
            sx_lgcrp = ss[argmax(va_lgcrp)[1]];
            #-------------------------
            αx_lp[seed_val]    = αs[argmax(va_lp)];
            αx_lgc[seed_val]   = αs[argmax(va_lgc  )[2]];
            αx_lgcrp[seed_val] = αs[argmax(va_lgcrp)[2]];
            αx_sgcrp[seed_val] = αs[argmax(va_sgcrp)];
            αx_gcnrp[seed_val] = αs[argmax(va_gcnrp)];
            #-------------------------

            #-------------------------
            # run on testing set
            #-------------------------
            ac_lr[seed_val]    = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="mlp",  residual_propagation=false,                                             σ=identity, n_step=sx_lr,    cb_skip=cb_skip, seed_val=seed_val);
            #-------------------------
            ac_sgc[seed_val]   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=false,                         K=Kx_sgc[seed_val], σ=identity, n_step=sx_sgc,   cb_skip=cb_skip, seed_val=seed_val);
            ac_gcn[seed_val]   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=false,                         K=Kx_gcn[seed_val], σ=relu,     n_step=sx_gcn,   cb_skip=cb_skip, seed_val=seed_val);
            #-------------------------
            ac_lp[seed_val]    = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="mean", residual_propagation=true,  α=αx_lp[seed_val],                                                                        seed_val=seed_val);
            ac_lgc[seed_val]   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=true,  predictor="mlp",  residual_propagation=false, α=αx_lgc[seed_val],                         σ=identity, n_step=sx_lgc,   cb_skip=cb_skip, seed_val=seed_val);
            ac_lgcrp[seed_val] = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=true,  predictor="mlp",  residual_propagation=true,  α=αx_lgcrp[seed_val],                       σ=identity, n_step=sx_lgcrp, cb_skip=cb_skip, seed_val=seed_val);
            ac_sgcrp[seed_val] = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=αx_sgcrp[seed_val],   K=Kx_sgc[seed_val], σ=identity, n_step=sx_sgc,   cb_skip=cb_skip, seed_val=seed_val);
            ac_gcnrp[seed_val] = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=αx_gcnrp[seed_val],   K=Kx_gcn[seed_val], σ=relu,     n_step=sx_gcn,   cb_skip=cb_skip, seed_val=seed_val);
            #-------------------------
        end

        if compute_VI
        @printf("%6.3f;       vi_ll; %s\n", split_ratio, array2str(vi_ll));
        @printf("%6.3f;       vi_ff; %s\n", split_ratio, array2str(vi_ff));
        @printf("%6.3f;       vi_fl; %s\n", split_ratio, array2str(vi_fl));
        end

        #-----------------------------
        @printf("%6.3f;       ac_lr; %s\n", split_ratio, array2str(ac_lr));
        #-----------------------------
        @printf("%6.3f;      ac_sgc; %s\n", split_ratio, array2str(ac_sgc));
        @printf("%6.3f;      ac_gcn; %s\n", split_ratio, array2str(ac_gcn));
        #-----------------------------
        @printf("%6.3f;       ac_lp; %s\n", split_ratio, array2str(ac_lp));
        @printf("%6.3f;      ac_lgc; %s\n", split_ratio, array2str(ac_lgc));
        @printf("%6.3f;    ac_lgcrp; %s\n", split_ratio, array2str(ac_lgcrp));
        @printf("%6.3f;    ac_sgcrp; %s\n", split_ratio, array2str(ac_sgcrp));
        @printf("%6.3f;    ac_gcnrp; %s\n", split_ratio, array2str(ac_gcnrp));
        #-----------------------------

        #-----------------------------
        @printf("%6.3f;      Kx_sgc; %s\n", split_ratio, array2str(Kx_sgc));
        @printf("%6.3f;      Kx_gcn; %s\n", split_ratio, array2str(Kx_gcn));
        #-----------------------------
        @printf("%6.3f;       αx_lp; %s\n", split_ratio, array2str(αx_lp));
        @printf("%6.3f;      αx_lgc; %s\n", split_ratio, array2str(αx_lgc));
        @printf("%6.3f;    αx_lgcrp; %s\n", split_ratio, array2str(αx_lgcrp));
        @printf("%6.3f;    αx_sgcrp; %s\n", split_ratio, array2str(αx_sgcrp));
        @printf("%6.3f;    αx_gcnrp; %s\n", split_ratio, array2str(αx_gcnrp));
        #-----------------------------

        #-----------------------------
        flush(stdout);
        #-----------------------------
    end
end

function run_inductive(G, labels, feats, G_new, labels_new, feats_new)
    """
    Args:
         shift: added to random initialization of correlation parameters
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    n_step = 500;
    cb_skip = 100;
    cv_fold = 5;
    #-----------------------------------------------------------------------------------
    ntrials = 10;
    #-----------------------------------------------------------------------------------
    ss = collect(cb_skip:cb_skip:n_step)
    Ks = [1, 2, 3];
    αs = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 0.999];
    #-----------------------------------------------------------------------------------
    nf = length(feats[1]);
    #-----------------------------------------------------------------------------------

    #-----------------------------
    ac_lr    = zeros(ntrials,2);
    #-----------------------------
    ac_sgc   = zeros(ntrials,2);
    ac_gcn   = zeros(ntrials,2);
    #-------------------------
    ac_lp    = zeros(ntrials,2);
    ac_lgc   = zeros(ntrials,2);
    ac_lgcrp = zeros(ntrials,2);
    ac_sgcrp = zeros(ntrials,2);
    ac_gcnrp = zeros(ntrials,2);
    #-------------------------
    Kx_sgc   = zeros(Int, ntrials);
    Kx_gcn   = zeros(Int, ntrials);
    #-------------------------
    αx_lp    = zeros(ntrials);
    αx_lgc   = zeros(ntrials);
    αx_lgcrp = zeros(ntrials);
    αx_sgcrp = zeros(ntrials);
    αx_gcnrp = zeros(ntrials);
    #-------------------------
    coeff_lr  = zeros(nf+1);
    coeff_sgc = zeros(nf+1);
    coeff_lgc = zeros(nf+1);
    #-------------------------

    split_ratio = 0.30;

    # this function is used to recover the linear regression coefficients
    # given a learned predmap function
    function lr_coeff(predmap)
        ev(i) = (vec = zeros(nf); (i > 0) && (vec[i] = 1.0); vec);

        β = [];

        for i in 0:nf
            push!(β, data((i == 0) ? predmap(Graph(1),ev(i),[1]) : predmap(Graph(1),ev(i),[1])-predmap(Graph(1),ev(0),[1]))[1]);
        end

        return β;
    end

    #-----------------------------
    for seed_val in 1:ntrials
        #-------------------------
        dtr, dte = rand_split(nv(G), [split_ratio, 1.0-split_ratio]; seed_val=seed_val);
        #-------------------------

        #-------------------------
        # split training data into a number of folds
        #-------------------------
        dVAs = rand_split(dtr, ones(cv_fold)/cv_fold; seed_val=seed_val);
        #-------------------------

        #-------------------------
        va_lr  = zeros(length(ss));
        #-------------------------
        va_sgc = zeros(length(ss), length(Ks));
        va_gcn = zeros(length(ss), length(Ks));
        #-------------------------
        for dVA in dVAs
            #-------------------------
            dTR = setdiff(dtr, dVA);
            #-------------------------

            #-------------------------
            # cross validation for LR
            #-------------------------
            va_lr[:]    += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="mlp", residual_propagation=false,      σ=identity, n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
            #-------------------------

            #-------------------------
            # cross validation for SGC, GCN
            #-------------------------
            for (i,K) in enumerate(Ks)
            va_sgc[:,i] += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn", residual_propagation=false, K=K, σ=identity, n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
            va_gcn[:,i] += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn", residual_propagation=false, K=K, σ=relu,     n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
            end
            #-------------------------
        end
        #-------------------------
        sx_lr  = ss[argmax(va_lr)];
        #-------------------------
        sx_sgc = ss[argmax(va_sgc)[1]];
        sx_gcn = ss[argmax(va_gcn )[1]];
        #-------------------------
        Kx_sgc[seed_val] = Ks[argmax(va_sgc)[2]];
        Kx_gcn[seed_val] = Ks[argmax(va_gcn)[2]];
        #-------------------------

        #-------------------------
        va_lp    = zeros(length(αs));
        va_lgc   = zeros(length(ss), length(αs));
        va_lgcrp = zeros(length(ss), length(αs));
        va_sgcrp = zeros(length(αs));
        va_gcnrp = zeros(length(αs));
        #-------------------------
        for dVA in dVAs
            #-------------------------
            dTR = setdiff(dtr, dVA);
            #-------------------------

            #-------------------------
            # cross validation for LP, LGC, LGC/RP, SGC/RP, GCN/RP
            #-------------------------
            for (i,α) in enumerate(αs)
            va_lp[i]      += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="mean", residual_propagation=true,  α=α,                                                                 seed_val=seed_val, return_trace=false);
            va_lgc[:,i]   += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=true,  predictor="mlp",  residual_propagation=false, α=α,                     σ=identity, n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
            va_lgcrp[:,i] += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=true,  predictor="mlp",  residual_propagation=true,  α=α,                     σ=identity, n_step=n_step, cb_skip=cb_skip, seed_val=seed_val, return_trace=true)[2];
            va_sgcrp[i]   += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=α, K=Kx_sgc[seed_val], σ=identity, n_step=sx_sgc, cb_skip=cb_skip, seed_val=seed_val, return_trace=false);
            va_gcnrp[i]   += run_dataset(G, feats, labels, dTR, dVA, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=α, K=Kx_gcn[seed_val], σ=relu,     n_step=sx_gcn, cb_skip=cb_skip, seed_val=seed_val, return_trace=false);
            end
            #-------------------------
        end
        #-------------------------
        sx_lgc   = ss[argmax(va_lgc  )[1]];
        sx_lgcrp = ss[argmax(va_lgcrp)[1]];
        #-------------------------
        αx_lp[seed_val]    = αs[argmax(va_lp)];
        αx_lgc[seed_val]   = αs[argmax(va_lgc  )[2]];
        αx_lgcrp[seed_val] = αs[argmax(va_lgcrp)[2]];
        αx_sgcrp[seed_val] = αs[argmax(va_sgcrp)];
        αx_gcnrp[seed_val] = αs[argmax(va_gcnrp)];
        #-------------------------

        #-------------------------
        # run on testing set
        #-------------------------
        ac_lr[seed_val,1],    predmap_lr    = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="mlp",  residual_propagation=false,                                             σ=identity, n_step=sx_lr,      cb_skip=cb_skip, seed_val=seed_val, return_predmap=true);
        #-------------------------
        ac_sgc[seed_val,1],   predmap_sgc   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=false,                         K=Kx_sgc[seed_val], σ=identity, n_step=sx_sgc,     cb_skip=cb_skip, seed_val=seed_val, return_predmap=true);
        ac_gcn[seed_val,1],   predmap_gcn   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=false,                         K=Kx_gcn[seed_val], σ=relu,     n_step=sx_gcn,     cb_skip=cb_skip, seed_val=seed_val, return_predmap=true);
        #-------------------------
        ac_lp[seed_val,1],    predmap_lp    = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="mean", residual_propagation=true,  α=αx_lp[seed_val],                                                                        seed_val=seed_val, return_predmap=true);
        ac_lgc[seed_val,1],   predmap_lgc   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=true,  predictor="mlp",  residual_propagation=false, α=αx_lgc[seed_val],                         σ=identity, n_step=sx_lgc,   cb_skip=cb_skip, seed_val=seed_val, return_predmap=true);
        ac_lgcrp[seed_val,1], predmap_lgcrp = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=true,  predictor="mlp",  residual_propagation=true,  α=αx_lgcrp[seed_val],                       σ=identity, n_step=sx_lgcrp, cb_skip=cb_skip, seed_val=seed_val, return_predmap=true);
        ac_sgcrp[seed_val,1], predmap_sgcrp = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=αx_sgcrp[seed_val],   K=Kx_sgc[seed_val], σ=identity, n_step=sx_sgc,   cb_skip=cb_skip, seed_val=seed_val, return_predmap=true);
        ac_gcnrp[seed_val,1], predmap_gcnrp = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=αx_gcnrp[seed_val],   K=Kx_gcn[seed_val], σ=relu,     n_step=sx_gcn,   cb_skip=cb_skip, seed_val=seed_val, return_predmap=true);
        #-------------------------

        #-------------------------
        # transfer learning
        #-------------------------
        ac_lr[seed_val,2]    = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=false, predictor=predmap_lr,    residual_propagation=false,                                             σ=identity, n_step=sx_lr,    cb_skip=cb_skip, seed_val=seed_val);
        #-------------------------
        ac_sgc[seed_val,2]   = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=false, predictor=predmap_sgc,   residual_propagation=false,                         K=Kx_sgc[seed_val], σ=identity, n_step=sx_sgc,   cb_skip=cb_skip, seed_val=seed_val);
        ac_gcn[seed_val,2]   = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=false, predictor=predmap_gcn,   residual_propagation=false,                         K=Kx_gcn[seed_val], σ=relu,     n_step=sx_gcn,   cb_skip=cb_skip, seed_val=seed_val);
        #-------------------------
        ac_lp[seed_val,2]    = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=false, predictor=predmap_lp,    residual_propagation=true,  α=αx_lp[seed_val],                                                                        seed_val=seed_val);
        ac_lgc[seed_val,2]   = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=true,  predictor=predmap_lgc,   residual_propagation=false, α=αx_lgc[seed_val],                         σ=identity, n_step=sx_lgc,   cb_skip=cb_skip, seed_val=seed_val);
        ac_lgcrp[seed_val,2] = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=true,  predictor=predmap_lgcrp, residual_propagation=true,  α=αx_lgcrp[seed_val],                       σ=identity, n_step=sx_lgcrp, cb_skip=cb_skip, seed_val=seed_val);
        ac_sgcrp[seed_val,2] = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=false, predictor=predmap_sgcrp, residual_propagation=true,  α=αx_sgcrp[seed_val],   K=Kx_sgc[seed_val], σ=identity, n_step=sx_sgc,   cb_skip=cb_skip, seed_val=seed_val);
        ac_gcnrp[seed_val,2] = run_dataset(G_new, feats_new, labels_new, dtr, dte, feature_smoothing=false, predictor=predmap_gcnrp, residual_propagation=true,  α=αx_gcnrp[seed_val],   K=Kx_gcn[seed_val], σ=relu,     n_step=sx_gcn,   cb_skip=cb_skip, seed_val=seed_val);
        #-------------------------

        #-------------------------
        coeff_lr[:]  += lr_coeff(predmap_lr);
        coeff_lgc[:] += lr_coeff(predmap_lgc);
        coeff_sgc[:] += lr_coeff(predmap_sgc);
        #-------------------------
    end

    #-----------------------------
    @printf("transductive;       ac_lr; %s\n", array2str(   ac_lr[:,1]));
    @printf("transductive;      ac_sgc; %s\n", array2str(  ac_sgc[:,1]));
    @printf("transductive;      ac_gcn; %s\n", array2str(  ac_gcn[:,1]));
    @printf("transductive;       ac_lp; %s\n", array2str(   ac_lp[:,1]));
    @printf("transductive;      ac_lgc; %s\n", array2str(  ac_lgc[:,1]));
    @printf("transductive;    ac_lgcrp; %s\n", array2str(ac_lgcrp[:,1]));
    @printf("transductive;    ac_sgcrp; %s\n", array2str(ac_sgcrp[:,1]));
    @printf("transductive;    ac_gcnrp; %s\n", array2str(ac_gcnrp[:,1]));
    #-----------------------------

    #-----------------------------
    @printf("   inductive;       ac_lr; %s\n", array2str(   ac_lr[:,2]));
    @printf("   inductive;      ac_sgc; %s\n", array2str(  ac_sgc[:,2]));
    @printf("   inductive;      ac_gcn; %s\n", array2str(  ac_gcn[:,2]));
    @printf("   inductive;       ac_lp; %s\n", array2str(   ac_lp[:,2]));
    @printf("   inductive;      ac_lgc; %s\n", array2str(  ac_lgc[:,2]));
    @printf("   inductive;    ac_lgcrp; %s\n", array2str(ac_lgcrp[:,2]));
    @printf("   inductive;    ac_sgcrp; %s\n", array2str(ac_sgcrp[:,2]));
    @printf("   inductive;    ac_gcnrp; %s\n", array2str(ac_gcnrp[:,2]));
    #-----------------------------

    #-----------------------------
    @printf("coefficients;         lr; %s\n", array2str(coeff_lr/ntrials));
    @printf("coefficients;        lgc; %s\n", array2str(coeff_lgc/ntrials));
    @printf("coefficients;        sgc; %s\n", array2str(coeff_sgc/ntrials));
    #-----------------------------
end
