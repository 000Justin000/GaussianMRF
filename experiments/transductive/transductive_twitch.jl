include("fit_gmrf.jl");
include("predict.jl");

function run_twitch(compute_VI=false)
    """
    Args:
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    dataset = "twitch_PTBR_true_64";
    G, _, labels, feats = read_network(dataset);
    #-----------------------------------------------------------------------------------
    if compute_VI
        ρ_string, α_string = split(readlines("results/coeff_"*dataset)[end], ';');
        ρ = parse.(Float64, split(match(r".*ρ_opt:(.+)", ρ_string)[1], ','));
        α = parse.(Float64, split(match(r".*α_opt:(.+)", α_string)[1], ','));
        p = length(feats[1]) + length(labels[1]);
        lidx = [p];
        fidx = collect(1:p-1);
    else
        α = nothing;
        lidx = nothing;
        fidx = nothing;
    end
    #-----------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------
    run_transductive(G, labels, feats; compute_VI=compute_VI, α=α, lidx=lidx, fidx=fidx);
    #-----------------------------------------------------------------------------------
end
