include("fit_gmrf.jl");
include("predict.jl");

function run_synthetic(graph_size="medium", p=5, lidx=[5], shift=0.0, sid=1, compute_VI=false)
    """
    Args:
         shift: added to random initialization of correlation parameters
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    dataset = "synthetic_" * graph_size * "_" * @sprintf("%+2.1f", shift) * "_" * string(sid);
    G, _, labels, feats = read_network(dataset * "_" * string(lidx[1]));
    #-----------------------------------------------------------------------------------
    if compute_VI
        ρ_string, ξ_string = split(readlines("results/coeff_"*dataset)[end], ';');
        ρ = parse.(Float64, split(match(r".*ρ_opt:(.+)", ρ_string)[1], ','));
        ξ = parse.(Float64, split(match(r".*ξ_opt:(.+)", ξ_string)[1], ','));
        fidx = setdiff(1:p, lidx);
    else
        ξ = nothing;
        lidx = nothing;
        fidx = nothing;
    end
    #-----------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------
    run_transductive(G, labels, feats; compute_VI=compute_VI, ξ=ξ, lidx=lidx, fidx=fidx);
    #-----------------------------------------------------------------------------------
end
