include("read_network.jl");
include("predict.jl");

G, _, labels, feats = read_network("bitcoin_transaction");

# random split
known_ids = collect(1:nv(G))[(!isnan).(labels)];
dtr, dva, dte = rand_split(known_ids, [31/49, 5/49, 13/49]);

# split data along time dimension
# TT = (x -> x[1]).(feats);
# UT = sort(unique(TT));
# TRT, VAT, TET = UT[1:31], UT[32:36], UT[37:49];
# dtr = collect(1:nv(G))[(x -> x in TRT).(TT) .& (!isnan).(labels)];
# dva = collect(1:nv(G))[(x -> x in VAT).(TT) .& (!isnan).(labels)];
# dte = collect(1:nv(G))[(x -> x in TET).(TT) .& (!isnan).(labels)];

n_step = 500;
αs = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 0.999];

#-------------------------
va_lr    = Vector(undef, 1);
va_sgc   = Vector(undef, 1);
va_gcn   = Vector(undef, 1);
#-------------------------
va_lp    = Vector(undef, length(αs));
va_lgc   = Vector(undef, length(αs));
va_lgcrp = Vector(undef, length(αs));
va_sgcrp = Vector(undef, length(αs));
va_gcnrp = Vector(undef, length(αs));
#-------------------------

#-------------------------
va_lr[1]  = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=false, predictor="mlp",  residual_propagation=false,      σ=identity, n_step=n_step, return_predictions=true);
va_sgc[1] = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=false, predictor="gcn",  residual_propagation=false,      σ=identity, n_step=n_step, return_predictions=true);
va_gcn[1] = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=false, predictor="gcn",  residual_propagation=false,      σ=relu,     n_step=n_step, return_predictions=true);
#-------------------------
for (i,α) in enumerate(αs)
va_lp[i]    = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=false, predictor="mean", residual_propagation=true,  α=α,             n_step=n_step, return_predictions=true);
va_lgc[i]   = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=true,  predictor="mlp",  residual_propagation=false, α=α, σ=identity, n_step=n_step, return_predictions=true);
va_lgcrp[i] = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=true,  predictor="mlp",  residual_propagation=true,  α=α, σ=identity, n_step=n_step, return_predictions=true);
va_sgcrp[i] = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=α, σ=identity, n_step=n_step, return_predictions=true);
va_gcnrp[i] = run_dataset(G, feats, labels, dtr, dva, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=α, σ=relu,     n_step=n_step, return_predictions=true);
end
#-------------------------

function optimal_F1(yy, thresh=nothing)
    y_, y = yy[1][:], Int.(yy[2][:]);
    thresh = (thresh == nothing) ? range(minimum(y_), stop=maximum(y_), length=100) : thresh;
    f1scores = f1score.(roc(y, y_, thresh));

    return maximum(f1scores), thresh[argmax(f1scores)];

end

te_lr    = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="mlp",  residual_propagation=false,                                                        σ=identity, n_step=n_step, return_trace=true, return_predictions=true);
te_sgc   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=false,                                                        σ=identity, n_step=n_step, return_trace=true, return_predictions=true);
te_gcn   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=false,                                                        σ=relu,     n_step=n_step, return_trace=true, return_predictions=true);

te_lp    = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="mean", residual_propagation=true,  α=αs[argmax((yy -> optimal_F1(yy)[1]).(va_lp))],                   n_step=n_step, return_trace=true, return_predictions=true);
te_lgc   = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=true,  predictor="mlp",  residual_propagation=false, α=αs[argmax((yy -> optimal_F1(yy)[1]).(va_lgc))],   σ=identity, n_step=n_step, return_trace=true, return_predictions=true);
te_lgcrp = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=true,  predictor="mlp",  residual_propagation=true,  α=αs[argmax((yy -> optimal_F1(yy)[1]).(va_lgcrp))], σ=identity, n_step=n_step, return_trace=true, return_predictions=true);
te_sgcrp = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=αs[argmax((yy -> optimal_F1(yy)[1]).(va_sgcrp))],   σ=identity, n_step=n_step, return_trace=true, return_predictions=true);
te_gcnrp = run_dataset(G, feats, labels, dtr, dte, feature_smoothing=false, predictor="gcn",  residual_propagation=true,  α=αs[argmax((yy -> optimal_F1(yy)[1]).(va_gcnrp))],    σ=relu,     n_step=n_step, return_trace=true, return_predictions=true);

final_F1(te_algo, va_algo) = optimal_F1(te_algo, [optimal_F1(va_algo[argmax((yy -> optimal_F1(yy)[1]).(va_algo))])[2]])[1];

@printf("          LR F1: %6.3f\n", final_F1(te_lr,  va_lr));
@printf("         SGC F1: %6.3f\n", final_F1(te_sgc, va_sgc));
@printf("         GCN F1: %6.3f\n", final_F1(te_gcn, va_gcn));

@printf("          LP F1: %6.3f\n", final_F1(te_lp,    va_lp));
@printf("         LGC F1: %6.3f\n", final_F1(te_lgc,   va_lgc));
@printf("      LGC/RP F1: %6.3f\n", final_F1(te_lgcrp, va_lgcrp));
@printf("      SGC/RP F1: %6.3f\n", final_F1(te_sgcrp, va_sgcrp));
@printf("      GCN/RP F1: %6.3f\n", final_F1(te_gcnrp, va_gcnrp));
