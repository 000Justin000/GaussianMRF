include("../../fit_gmrf.jl");
include("../../predict.jl");

function run_synthetic(sid_tr, sid_te, graph_type="WattsStrogatzOriginal", lidx=[5], shift=0.0)
    """
    Args:
         shift: added to random initialization of correlation parameters
    """
    #-----------------------------------------------------------------------------------
    dataset = "synthetic_" * graph_type * "_" * @sprintf("%+2.1f", shift)
    #-----------------------------------------------------------------------------------
    G,     _, labels,     feats     = read_network(dataset * @sprintf("_%1d_", sid_tr) * string(lidx[1]));
    G_new, _, labels_new, feats_new = read_network(dataset * @sprintf("_%1d_", sid_te) * string(lidx[1]));
    #-----------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------
    run_inductive(G, labels, feats, G_new, labels_new, feats_new);
    #-----------------------------------------------------------------------------------
end
