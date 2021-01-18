include("fit_gmrf.jl");
include("predict.jl");

function run_climate(year=2008, lb="airT", compute_VI=false)
    """
    Args:
            lb: specify which attribute is being predicted
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    dataset = "climate_" * string(year);
    G, _, labels, feats = read_network(dataset * "_" * lb);
    #-----------------------------------------------------------------------------------
    if compute_VI
        ρ_string, α_string = split(readlines("results/coeff_"*dataset)[end], ';');
        ρ = parse.(Float64, split(match(r".*ρ_opt:(.+)", ρ_string)[1], ','));
        α = parse.(Float64, split(match(r".*α_opt:(.+)", α_string)[1], ','));
        lb2idx = Dict("airT"=>1, "landT"=>2, "precipitation"=>3, "sunlight"=>4, "pm2.5"=>5);
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
