include("../../fit_gmrf.jl");
include("../../predict.jl");

function run_county_facebook(year_tr, year_te, lb="income")
    """
    Args:
            lb: specify which attribute is being predictedn
    compute_VI: whether to compute the mutual information between the labels and the predictors
    """
    #-----------------------------------------------------------------------------------
    G,     _, labels,     feats     = read_network("county_facebook_" * @sprintf("%4d_", year_tr) * lb);
    G_new, _, labels_new, feats_new = read_network("county_facebook_" * @sprintf("%4d_", year_te) * lb);
    #-----------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------
    run_inductive(G, labels, feats, G_new, labels_new, feats_new);
    #-----------------------------------------------------------------------------------
end
