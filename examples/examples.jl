include("read_network.jl");
include("predict.jl");

# read the four requirements as listed above
G, _, labels, feats = read_network("county_facebook_2016_election");

# random split, 30% training and 70% testing
ll, uu = rand_split(nv(G), [0.3, 0.7]);

# prediction routine
run_dataset(G, feats, labels, ll, uu; feature_smoothing=true, α=0.95, predictor="mlp", σ=identity, residual_propagation=true)
