using LightGraphs;
using StatsBase;
using Statistics;
using DelimitedFiles;
using DataFrames;
using Printf;
using CSV;
using JSON;
using Flux;
using Arpack;
using LinearAlgebra;
using SparseArrays;
using MultivariateStats;

max_normalize(x) = maximum(abs.(x)) == 0 ? x : x/maximum(abs.(x));
std_normalize(x) = std(x) == 0 ? zeros(length(x)) : (x.-mean(x))./std(x);
int_normalize(x) = std(x) == 0 ? zeros(length(x)) : (x.-minimum(x))/(maximum(x).-minimum(x))*2 .- 1;

function parse_mean_fill(vr, normalize=false)
    vb = mean(map(x->(typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), filter(!ismissing, vr)));
    vv = collect(map(x->ismissing(x) ? vb : (typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), vr));
    if normalize
        vv = (vv .- vb) / std(vv);
    end
    return vv;
end

function read_synthetic(graph_type="WattsStrogatz", shift=0.0, sample_id=1, prediction_id=1)
    dat = JSON.parsefile("datasets/synthetic/" * graph_type * "/shift_" * @sprintf("%+2.1f", shift) * ".json");

    g = Graph([dat["A"][j][i] for i = 1:length(dat["A"][1]), j = 1:length(dat["A"])]);
    Y = [dat["Y"][k][j][i] for i = 1:length(dat["Y"][1][1]), j = 1:length(dat["Y"][1]), k = 1:length(dat["Y"])];

    ff = f32(Y[:,:,sample_id]');
    pos = prediction_id;

    y = ff[:,pos];
    f = [vcat(ff[i,1:pos-1], ff[i,pos+1:end]) for i in 1:size(ff,1)];

    return g, [adjacency_matrix(g)], y, f;
end

function read_county_facebook(year, prediction)
    VOT = CSV.read("datasets/county_facebook/election.csv", DataFrame);
    ICM = CSV.read("datasets/county_facebook/income.csv", DataFrame);
    POP = CSV.read("datasets/county_facebook/population.csv", DataFrame);
    EDU = CSV.read("datasets/county_facebook/education.csv", DataFrame);
    UEP = CSV.read("datasets/county_facebook/unemployment.csv", DataFrame);

    vot = DataFrames.DataFrame((:FIPS=>VOT[:,:fips_code], :DEM=>VOT[:,Symbol("dem_", year)], :GOP=>VOT[:,Symbol("gop_", year)]));
    icm = DataFrames.DataFrame((:FIPS=>ICM[:,:FIPS], :MedianIncome=>ICM[:,Symbol("MedianIncome", min(max(2011,year), 2018))]));
    pop = DataFrames.DataFrame((:FIPS=>POP[:,:FIPS], :MigraRate=>POP[:,Symbol("R_NET_MIG_", min(max(2011,year), 2018))],
                                                     :BirthRate=>POP[:,Symbol("R_birth_", min(max(2011,year), 2018))],
                                                     :DeathRate=>POP[:,Symbol("R_death_", min(max(2011,year), 2018))]));
    edu = DataFrames.DataFrame((:FIPS=>EDU[:,Symbol("FIPS")], :BachelorRate=>EDU[:,Symbol("BachelorRate", year)]));
    uep = DataFrames.DataFrame((:FIPS=>UEP[:,:FIPS], :UnemploymentRate=>UEP[:,Symbol("Unemployment_rate_", min(max(2007,year), 2018))]));

    df = CSV.read("datasets/county_facebook/county-info.csv", DataFrame; header=1, types=[Int,Int,Float64,Float64,Float64], strict=false, silencewarnings=true);
    info = rename(df[completecases(df),:], [:FIPS, :id, :sh050m, :sh100m, :sh500m]);
    
    # change of FIPS code of Oglala Lakota County (FIPS 46102) to the old Shannon County (FIPS 46113)
    info[findfirst(info[!,:FIPS] .== 46102), :FIPS] = 46113;
    pop[findfirst(pop[!,:FIPS] .== 46102), :FIPS] = 46113;
    uep[findfirst(uep[!,:FIPS] .== 46102), :FIPS] = 46113;

    jfl(df1, df2) = innerjoin(df1, df2, on=:FIPS);
    dat = sort(jfl(jfl(jfl(jfl(jfl(info, icm), pop), edu), uep), vot), [:FIPS]);

    id2num = Dict{Int,Int}(id=>num for (num,id) in enumerate(dat[!,2]));

    adj = Int.(Matrix(CSV.read("datasets/county_facebook/top.csv", DataFrame)[!,1:2]));
    n = length(id2num);

    G = Graph(n);
    for (h,t) in zip(adj[:,1],adj[:,2])
        if (h != t) && haskey(id2num,h) && haskey(id2num,t)
            add_edge!(G, id2num[h], id2num[t]);
        end
    end

    # extract features and label
    ff = zeros(Float32, size(dat,1),10);
    for i in 1:9
        ff[:,i] = parse_mean_fill(dat[:,i+2], true);
    end
    dem = parse_mean_fill(dat[:,12]);
    gop = parse_mean_fill(dat[:,13]);
    ff[:,10] = parse_mean_fill((gop-dem)./(gop+dem), true);

    if prediction == "sh050m"
        pos = 1;
    elseif prediction == "sh100m"
        pos = 2;
    elseif prediction == "sh500m"
        pos = 3;
    elseif prediction == "income"
        pos = 4;
    elseif prediction == "migration"
        pos = 5;
    elseif prediction == "birth"
        pos = 6;
    elseif prediction == "death"
        pos = 7;
    elseif prediction == "education"
        pos = 8;
    elseif prediction == "unemployment"
        pos = 9;
    elseif prediction == "election"
        pos = 10;
    else
        error("unexpected prediction type");
    end

    y = ff[:,pos];
    f = [vcat(ff[i,1:pos-1], ff[i,pos+1:end]) for i in 1:size(ff,1)];

    return G, [adjacency_matrix(G)], y, f;
end

function read_climate(year, prediction)
    adj = CSV.read("datasets/climate/adjacency.txt", DataFrame; header=0);
    fips2cty = Dict();
    for i in 1:size(adj,1)
        if !ismissing(adj[i,2])
            fips2cty[adj[i,2]] = adj[i,1];
        end
    end

    hh = adj[:,2];
    tt = adj[:,4];

    @assert !ismissing(hh[1]);
    for i in 2:size(hh,1)
        ismissing(hh[i]) && (hh[i] = hh[i-1]);
    end
    hh = convert(Vector{Int}, hh);

    fips = sort(unique(union(hh,tt)));
    id2num = Dict(id=>num for (num,id) in enumerate(fips));
    G = Graph(length(id2num));
    for (h,t) in zip(hh,tt)
        (h != t) && add_edge!(G, id2num[h], id2num[t]);
    end

    MAT = filter(x -> x[:Year] == year, CSV.read("datasets/climate/max_air_temperature.txt", DataFrame));
    LST = filter(x -> x[:Year] == year, CSV.read("datasets/climate/land_surface_temperature.txt", DataFrame));
    PCP = filter(x -> x[:Year] == year, CSV.read("datasets/climate/percipitation.txt", DataFrame));
    SLT = filter(x -> x[:Year] == year, CSV.read("datasets/climate/sunlight.txt", DataFrame));
    FPM = filter(x -> x[:Year] == year, CSV.read("datasets/climate/fine_particulate_matter.txt", DataFrame));

    cty = DataFrames.DataFrame((:FIPS=>fips, :County=>[fips2cty[fips_] for fips_ in fips]));
    mat = DataFrames.DataFrame((:FIPS=>MAT[:,Symbol("County Code")], :MAT=>MAT[:,Symbol("Avg Daily Max Air Temperature (C)")]));
    lst = DataFrames.DataFrame((:FIPS=>LST[:,Symbol("County Code")], :LST=>LST[:,Symbol("Avg Day Land Surface Temperature (C)")]));
    pcp = DataFrames.DataFrame((:FIPS=>PCP[:,Symbol("County Code")], :PCP=>PCP[:,Symbol("Avg Daily Precipitation (mm)")]));
    slt = DataFrames.DataFrame((:FIPS=>SLT[:,Symbol("County Code")], :SLT=>SLT[:,Symbol("Avg Daily Sunlight (KJ/m\xb2)")]));
    fpm = DataFrames.DataFrame((:FIPS=>FPM[:,Symbol("County Code")], :FPM=>FPM[:,Symbol("Avg Fine Particulate Matter (\xb5g/m\xb3)")]));

    jfl(df1, df2) = innerjoin(df1, df2, on=:FIPS);
    dat = sort(jfl(jfl(jfl(jfl(jfl(cty, mat), lst), pcp), slt), fpm), [:FIPS]);
    g, ori_id = induced_subgraph(G, [id2num[id] for id in dat[:,:FIPS]]);

    ff = zeros(Float32, size(dat,1), 5);
    for i in 1:5
        ff[:,i] = parse_mean_fill(dat[:,i+2], true);
    end

    if prediction == "airT"
        pos = 1;
    elseif prediction == "landT"
        pos = 2;
    elseif prediction == "precipitation"
        pos = 3;
    elseif prediction == "sunlight"
        pos = 4;
    elseif prediction == "pm2.5"
        pos = 5;
    else
        error("unexpected prediction type");
    end

    y = ff[:,pos];
    f = [vcat(ff[i,1:pos-1], ff[i,pos+1:end]) for i in 1:size(ff,1)];

    return g, [adjacency_matrix(g)], y, f;
end

function read_ward(year, prediction)
    code = CSV.read("datasets/london/code.csv", DataFrame; header=1);
    W = CSV.read("datasets/london/W.csv", DataFrame; header=1)
    dat = CSV.read("datasets/london/MaycoordLondon.csv", DataFrame; header=1);

    info = leftjoin(code[:,2:end], dat, on=names(code)[2] => names(dat)[1]);

    indices = (!ismissing).(info[:,2]);
    W0 = Matrix(W[:,2:end] .!== 0.0)[indices, indices];
    info0 = info[indices,:];

    function parse_mean_fill(vr, normalize=false)
        vb = mean(map(x->(typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), filter(!ismissing, vr)));
        vv = collect(map(x->ismissing(x) ? vb : (typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), vr));
        if normalize
            vv = (vv .- vb) / std(vv);
        end
        return vv;
    end

    col = [5, 6, 7, 10, 11];
    ff = zeros(Float32, size(info0,1),6);
    for i in 1:5
        ff[:,i] = parse_mean_fill(info0[:,col[i]], true);
    end

    if year == 2016
        ff[:,6] = parse_mean_fill(info0[:,3], true);
    elseif year == 2012
        ff[:,6] = parse_mean_fill(info0[:,4], true);
    else
        error("unexpected year");
    end

    if prediction == "edu"
        pos = 1;
    elseif prediction == "age"
        pos = 2;
    elseif prediction == "gender"
        pos = 3;
    elseif prediction == "income"
        pos = 4;
    elseif prediction == "populationsize"
        pos = 5;
    elseif prediction == "election"
        pos = 6;
    else
        error("unexpected prediction type");
    end

    y = ff[:,pos];
    f = [vcat(ff[i,1:pos-1], ff[i,pos+1:end]) for i in 1:size(ff,1)];

    return Graph(W0), [float(W0)], y, f;
end

function read_twitch(cnm, dim_reduction=false, dim_embed=8)
    #----------------------------------------------------------------------------
    feats_all = [];
    for cn in ["DE", "ENGB", "ES", "FR", "PTBR", "RU"]
        feats = JSON.parsefile("datasets/twitch/" * cn * "/musae_" * cn * "_features.json");
        append!(feats_all, values(feats));
    end
    #----------------------------------------------------------------------------
    ndim = maximum(vcat(feats_all...)) + 1;
    #----------------------------------------------------------------------------
    function feat_encode(feat_list)
        vv = zeros(Float32, ndim);
        vv[feat_list .+ 1] .= 1.0;
        return vv;
    end
    #----------------------------------------------------------------------------
    f_all = feat_encode.(feats_all);
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    feats = JSON.parsefile("datasets/twitch/" * cnm * "/musae_" * cnm * "_features.json");
    id2ft = Dict(id+1=>ft for (id,ft) in zip(parse.(Int,keys(feats)), values(feats))); n = length(id2ft);
    @assert minimum(keys(id2ft)) == 1 && maximum(keys(id2ft)) == n;
    #----------------------------------------------------------------------------
    ff = [feat_encode(id2ft[id]) for id in sort(collect(keys(id2ft)))];
    #----------------------------------------------------------------------------
    g = Graph(length(ff));
    #----------------------------------------------------------------------------
    links = CSV.read("datasets/twitch/" * cnm * "/musae_" * cnm * "_edges.csv", DataFrame);
    for i in 1:size(links,1)
        add_edge!(g, links[i,:from]+1, links[i,:to]+1);
    end
    #----------------------------------------------------------------------------

    if dim_reduction
        U,S,V = svds(hcat(ff...); nsv=dim_embed)[1];
        UU = U .* sign.(sum(U,dims=1)[:])';
        f0 = [vcat(UU'*f_,d_) for (f_,d_) in zip(ff,std_normalize(sqrt.(degree(g))))];
        fbar = mean(f0);
        f = [f_ - fbar for f_ in f0];
    end

    trgts = CSV.read("datasets/twitch/" * cnm * "/musae_" * cnm * "_target.csv", DataFrame);
    nid2days = Dict(zip(trgts[!,:new_id], trgts[!,:days]));
    y = std_normalize([nid2days[i-1] for i in 1:nv(g)]);

    return g, [adjacency_matrix(g)], y, f;
end

function read_bitcoin_transaction()
    feats = sort(CSV.read("datasets/elliptic_bitcoin/elliptic_txs_features.csv", DataFrame, header=0), [:Column1]);
    labels = sort(CSV.read("datasets/elliptic_bitcoin/elliptic_txs_classes.csv", DataFrame, header=1), [:txId]);
    adj = CSV.read("datasets/elliptic_bitcoin/elliptic_txs_edgelist.csv", DataFrame; header=1);

    hh = collect(adj[!,:txId1]);
    tt = collect(adj[!,:txId2]);
    txids = sort(unique(union(hh,tt)));
    @assert all(collect(feats[!,:Column1]) .== txids);

    id2num = Dict(id=>num for (num,id) in enumerate(txids));

    G = Graph(length(id2num));
    for (h,t) in zip(hh,tt)
        (h != t) && add_edge!(G, id2num[h], id2num[t]);
    end

    ff = zeros(Float32, nv(G),166);
    for i in 1:166
        ff[:,i] = std_normalize(feats[!,i+1]);
    end

    f = [ff[i,:] for i in 1:size(ff,1)];
    y = (x -> (x=="1") ? 1.0 : ((x=="2") ? 0.0 : NaN)).(labels[!,:class]);

    return G, [adjacency_matrix(G)], y, f;
end

function read_network(network_name)
    (p = match(r"^synthetic_([a-z]+)_([0-9.+-]+)_([0-9]+)_([0-9]+)$", network_name)) !== nothing && return read_synthetic(p[1], parse(Float64, p[2]), parse(Int64, p[3]), parse(Int64, p[4]));
    (p = match(r"^county_facebook_([0-9]+)_(.+)$", network_name)) !== nothing && return read_county_facebook(parse(Int, p[1]), p[2]);
    (p = match(r"^climate_([0-9]+)_(.+)$", network_name)) !== nothing && return read_climate(parse(Int, p[1]), p[2]);
    (p = match(r"^ward_([0-9]+)_(.+)$", network_name)) !== nothing && return read_ward(parse(Int, p[1]), p[2]);
    (p = match(r"^twitch_([0-9a-zA-Z]+)_([a-z]+)_([0-9]+)$", network_name)) !== nothing && return read_twitch(p[1], parse(Bool, p[2]), parse(Int, p[3]));
    (p = match(r"^bitcoin_transaction", network_name)) !== nothing && return read_bitcoin_transaction();
end
