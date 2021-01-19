include("fit_gmrf.jl");
include("predict.jl");

function run_ward(year=2016, lb="edu", compute_VI=false)
    """
    Args:
            lb: specify which attribute is being predictedn
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    dataset = "ward_" * string(year);
    G, _, labels, feats = read_network(dataset * "_" * lb);
    #-----------------------------------------------------------------------------------
    if compute_VI
        ρ_string, ξ_string = split(readlines("results/coeff_"*dataset)[end], ';');
        ρ = parse.(Float64, split(match(r".*ρ_opt:(.+)", ρ_string)[1], ','));
        ξ = parse.(Float64, split(match(r".*ξ_opt:(.+)", ξ_string)[1], ','));
        lb2idx = Dict("edu"=>1, "age"=>2, "gender"=>3, "income"=>4, "populationsize"=>5, "election"=>6);
        p = length(feats[1]) + length(labels[1]);
        lidx = [lb2idx[lb]];
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
