using CSV;
using DataFrames;
using Statistics;
using Plots; pyplot();
using LaTeXStrings;
using Printf;

function read_transductive(df, split_ratio, info)
    return parse.(Float64, strip.(split(filter(row->(row[:Column1]==split_ratio) && (strip(row[:Column2])==info), df)[1,:Column3], ',')));
end

function read_inductive(df, setting, info)
    return parse.(Float64, strip.(split(filter(row->(strip(row[:Column1])==setting) && (strip(row[:Column2])==info), df)[1,:Column3], ',')));
end

dataset = "synthetic_DegreeCorrectedStochasticBlockModel";
shift = "+2.0";
labels = ["1", "2", "3", "4", "5"];
label2id = Dict(label=>lid for (lid,label) in enumerate(labels));

algos   = ["LR", "LGC", "SGC", "GCN"];
algo2ac = Dict("LR"=>"ac_lr", "LP"=>"ac_lp", "LGC"=>"ac_lgc",      "SGC"=>"ac_sgc",      "GCN"=>"ac_gcn",      "SAGE"=>"ac_sgm", 
                                             "LGC/RP"=>"ac_lgcrp", "SGC/RP"=>"ac_sgcrp", "GCN/RP"=>"ac_gcnrp", "SAGE/RP"=>"ac_sagerp");

#-----------------------------
avg_ac_trs = zeros(length(labels), length(algos));
std_ac_trs = zeros(length(labels), length(algos));
#-----------------------------
for (i,label) in enumerate(labels)
    #-----------------------------
    for (j,algo) in enumerate(algos)
    #-----------------------------
        df = CSV.read("../results/transductive_"*dataset*"_"*shift*"_1_"*label, DataFrame; datarow=1, header=0, delim=';');
        
        avg_ac_trs[i,j] = mean(read_transductive(df, 0.3, algo2ac[algo]))
        std_ac_trs[i,j] =  std(read_transductive(df, 0.3, algo2ac[algo]))
    #-----------------------------
    end
    #-------------------------
end
#-----------------------------

#-----------------------------
avg_ac_ind = zeros(length(labels), length(algos));
std_ac_ind = zeros(length(labels), length(algos));
#-------------------------
for (i,label) in enumerate(labels)
    #-----------------------------
    for (j,algo) in enumerate(algos)
    #-----------------------------
        df = CSV.read("../results/inductive_"*dataset*"_"*shift*"_2_1_"*label, DataFrame; datarow=1, header=0, delim=';');
        
        avg_ac_ind[i,j] = mean(read_inductive(df, "inductive", algo2ac[algo]))
        std_ac_ind[i,j] =  std(read_inductive(df, "inductive", algo2ac[algo]))
    #-----------------------------
    end
    #-------------------------
end
#-----------------------------


bar_width = 0.15;
markersize = bar_width*100;
xx = 1:length(algos);
xx_trs = xx .- 0.9*bar_width;
xx_ind = xx .+ 0.9*bar_width;
yy_trs = mean(avg_ac_trs,dims=1)';
yy_ind = mean(avg_ac_ind,dims=1)';

h = plot(size=(550,600), xlim=(0.5,4.5), ylim=(0.0,1.0), xticks=(xx, algos), yticks=0.0:0.2:1.0, xtickfontsize=20, ytickfontsize=18, legendfontsize=18, framestyle=:box, grid=nothing);
bar!(h, xx_trs, yy_trs, bar_width=bar_width, linewidth=0, color=1, fillalpha=0.3, label=nothing);
bar!(h, xx_ind, yy_ind, bar_width=bar_width, linewidth=0, color=2, fillalpha=0.3, label=nothing);
scatter!(h, xx_trs, yy_trs, markersize=markersize, markerstrokewidth=0.0, color=1, label="transductive");
scatter!(h, xx_ind, yy_ind, markersize=markersize, markerstrokewidth=0.0, color=2, label="inductive");

for (xpos, ypos) in zip(xx_trs, yy_trs)
    annotate!(h, xpos-0.10, ypos+0.04, @sprintf("%3.2f", ypos));
end

for (xpos, ypos) in zip(xx_ind, yy_ind)
    annotate!(h, xpos+0.11, ypos+0.04, @sprintf("%3.2f", ypos));
end

annotate!(h, 0.70, 0.95, text(latexstring(@sprintf("h_{0} = %d", 10^Int(parse(Float64,shift)))), :left, 21));

savefig(h, "../figs/inductive_"*dataset*"_"*shift*".pdf");
savefig(h, "../figs/inductive_"*dataset*"_"*shift*".svg");
