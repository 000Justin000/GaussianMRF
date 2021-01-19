include("fit_gmrf.jl");
include("predict.jl");

function demo_synthetic(graph_size="medium")
    #-----------------------------------------------------------------------------------
    ωs = 10.0 .^ collect(-1.00:0.25:2.00);
    #-----------------------------------------------------------------------------------
    split_ratio = 0.3;
    ntrials = 30;
    #-----------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------
    ac = zeros(length(ωs), length(ωs), ntrials);
    #-----------------------------------------------------------------------------------
    for (i,ω0) in enumerate(ωs)
    #-----------------------------------------------------------------------------------
        Random.seed!(0);
        #-------------------------------------------------------------------------------
        G, _, Y = sample_synthetic(graph_size, nothing; synthetic_dict=Dict("p"=>2, "N"=>ntrials, "ξ0"=>[ω0,ω0,1.0,1.0,-0.99]));
        #-------------------------------------------------------------------------------
        for (j,ω) in enumerate(ωs)
        #-------------------------------------------------------------------------------
            for seed_val in 1:ntrials
                feats = [Y[1:1,i,seed_val] for i in 1:size(Y,2)];
                labels = Y[2,:,seed_val];
                #-------------------------------------------------------------------------------
                dtr, _, dte = rand_split(nv(G), [split_ratio, 0.0, 1.0-split_ratio]; seed_val=seed_val);
                ac[i,j,seed_val] = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=true, predictor="mlp", residual_propagation=true, σ=identity, α=ω/(1.0+ω), n_step=300, seed_val=seed_val)[end];
            end
        #-------------------------------------------------------------------------------
        end
        #-------------------------------------------------------------------------------
    end
    #-----------------------------------------------------------------------------------

    return mean(ac, dims=3)[:,:,1];
end

# ac_lgcrp = demo_synthetic("large");
# writedlm("results/lgcrp_accuracy.csv", ac_lgcrp);
ac_lgcrp = readdlm("results/lgcrp_accuracy.csv");

using Plots; pyplot();
using LaTeXStrings;

cmap = cgrad(:balance);
min_value = minimum(ac_lgcrp);
max_value = maximum(ac_lgcrp);
mid = (min_value < -0.1) ? (-min_value)/(max_value-min_value) : 0.5;
cmap.values[:] = vcat(range(0,stop=mid,length=15), range(mid,stop=1.00,length=16)[2:end]);

h = Plots.plot(size=(320,270), title="LGC/RP accuracy (" * L"R^{2}" * ")", xlabel="Gaussian MRF "*L"\omega", ylabel="LGC/RP "*L"\omega", framestyle=:box);
xx = collect(-1.00:0.25:2.00);
yy = collect(-1.00:0.25:2.00);
Plots.contour!(h, xx, yy, ac_lgcrp', fill=true, seriescolor=cmap, levels=20,
                  xticks=(xx[1:4:end], [L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]), 
                  yticks=(yy[1:4:end], [L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]));
Plots.plot!(h, xx, yy, linestyle=:solid, linewidth=1.5, color=:yellow, label=nothing);
Plots.scatter!(h, xx, yy[[idx[2] for idx in argmax(ac_lgcrp, dims=2)]], markersize=6.0, markercolor=:yellow, label=nothing);
Plots.savefig(h, "figs/lgcrp_accuracy.pdf");
Plots.savefig(h, "figs/lgcrp_accuracy.svg");
