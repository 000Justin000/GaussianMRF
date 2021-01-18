include("fit_gmrf.jl");
include("predict.jl");

function run_county_facebook(year=2016, lb="income", compute_VI=false)
    """
    Args:
            lb: specify which attribute is being predicted
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    dataset = "county_facebook_" * string(year);
    G, _, labels, feats = read_network(dataset * "_" * lb);
    #-----------------------------------------------------------------------------------
    if compute_VI
        ρ_string, α_string = split(readlines("results/coeff_"*dataset)[end], ';');
        ρ = parse.(Float64, split(match(r".*ρ_opt:(.+)", ρ_string)[1], ','));
        α = parse.(Float64, split(match(r".*α_opt:(.+)", α_string)[1], ','));
        lb2idx = Dict("sh050m"=>1, "sh100m"=>2, "sh500m"=>3, "income"=>4, "migration"=>5, "birth"=>6, "death"=>7, "education"=>8, "unemployment"=>9, "election"=>10);
        p = length(feats[1]) + length(labels[1]);
        lidx = [lb2idx[lb]];
        fidx = setdiff(1:p, lidx);
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
