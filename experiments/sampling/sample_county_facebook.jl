include("fit_gmrf.jl");
include("predict.jl");
include("extras.jl");

#-----------------------------------------------------------------------------------
function read_county_facebook_extras(year, prediction)
    VOT = CSV.read("datasets/county/election.csv", DataFrame);
    ICM = CSV.read("datasets/county/income.csv", DataFrame);
    POP = CSV.read("datasets/county/population.csv", DataFrame);
    EDU = CSV.read("datasets/county/education.csv", DataFrame);
    UEP = CSV.read("datasets/county/unemployment.csv", DataFrame);

    vot = DataFrames.DataFrame((:FIPS=>VOT[:,:fips_code], :DEM=>VOT[:,Symbol("dem_", year)], :GOP=>VOT[:,Symbol("gop_", year)]));
    icm = DataFrames.DataFrame((:FIPS=>ICM[:,:FIPS], :MedianIncome=>ICM[:,Symbol("MedianIncome", min(max(2011,year), 2018))]));
    pop = DataFrames.DataFrame((:FIPS=>POP[:,:FIPS], :MigraRate=>POP[:,Symbol("R_NET_MIG_", min(max(2011,year), 2018))],
                                                     :BirthRate=>POP[:,Symbol("R_birth_", min(max(2011,year), 2018))],
                                                     :DeathRate=>POP[:,Symbol("R_death_", min(max(2011,year), 2018))]));
    edu = DataFrames.DataFrame((:FIPS=>EDU[:,Symbol("FIPS")], :BachelorRate=>EDU[:,Symbol("BachelorRate", year)]));
    uep = DataFrames.DataFrame((:FIPS=>UEP[:,:FIPS], :UnemploymentRate=>UEP[:,Symbol("Unemployment_rate_", min(max(2007,year), 2018))]));

    df = CSV.read("datasets/more-graphs/facebook-counties/county-info.csv", DataFrame; header=1, types=[Int,Int,Float64,Float64,Float64], strict=false, silencewarnings=true);
    info = rename(df[completecases(df),:], [:FIPS, :id, :sh050m, :sh100m, :sh500m]);

    # change of FIPS code of Oglala Lakota County (FIPS 46102) to the old Shannon County (FIPS 46113)
    info[findfirst(info[!,:FIPS] .== 46102), :FIPS] = 46113;
    pop[findfirst(pop[!,:FIPS] .== 46102), :FIPS] = 46113;
    uep[findfirst(uep[!,:FIPS] .== 46102), :FIPS] = 46113;

    jfl(df1, df2) = innerjoin(df1, df2, on=:FIPS);
    dat = sort(jfl(jfl(jfl(jfl(jfl(info, icm), pop), edu), uep), vot), [:FIPS]);
    fips = dat[:,1];

    fips2id = Dict{Int,Int}(dat[i,:FIPS] => dat[i,:id] for i in 1:size(dat,1));
    id2num = Dict{Int,Int}(id=>num for (num,id) in enumerate(dat[!,2]));

    adj = Int.(Matrix(CSV.read("datasets/more-graphs/facebook-counties/top.csv", DataFrame)[!,1:2]));
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
        ff[:,i] = parse_mean_fill(dat[:,i+2]);
    end
    dem = parse_mean_fill(dat[:,12]);
    gop = parse_mean_fill(dat[:,13]);
    ff[:,10] = parse_mean_fill((gop-dem)./(gop+dem));

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

    recover_funcs = [];
    for i in 1:10
        μ,σ = mean(ff[:,i]), std(ff[:,i]);
        push!(recover_funcs, x -> x*σ+μ);
    end

    for i in 1:10
        ff[:,i] = std_normalize(ff[:,i]);
    end

    return G, [adjacency_matrix(G)], ff, fips, recover_funcs;
end
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
year = 2016;
lb = ["unemployment", "election"];
#-----------------------------------------------------------------------------------
G, _, ff, fips, recover_funcs = read_county_facebook_extras(year, "election");
#-----------------------------------------------------------------------------------
ρ_string, ξ_string = split(readlines("results/coeff_county_facebook_" * string(year))[end], ';');
ρ = parse.(Float64, split(match(r".*ρ_opt:(.+)", ρ_string)[1], ','));
ξ = parse.(Float64, split(match(r".*ξ_opt:(.+)", ξ_string)[1], ','));
lb2idx = Dict("sh050m"=>1, "sh100m"=>2, "sh500m"=>3, "income"=>4, "migration"=>5, "birth"=>6, "death"=>7, "education"=>8, "unemployment"=>9, "election"=>10);
p = size(ff,2);
lidx = [lb2idx[lb_] for lb_ in lb];
fidx = setdiff(1:p, lidx);
#-----------------------------------------------------------------------------------
interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]);
A = get_adjacency_matrices(G, p; interaction_list=interaction_list);
Γ = getΓ(ξ; A=A);
@assert isposdef(Γ);
#-----------------------------------------------------------------------------------
# the indices for features in fidx and vertices in V
FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
#-----------------------------------------------------------------------------------
L = FIDX(fidx);
U = FIDX(lidx);
#-----------------------------------------------------------------------------------
fl = ff[:,fidx];
fu = ff[:,lidx];
fe = reshape(cg(Γ[U,U], -Γ[U,L]*fl'[:]), (2,nv(G)))';
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
Σ = (M -> (M+M')/2)(inv(Matrix(Γ[U,U])));
g = MvNormal(Σ);
fss = [fe + reshape(rand(g), (2,nv(G)))' for _ in 1:10];
#-----------------------------------------------------------------------------------

plot_county(Dict(zip(fips,recover_funcs[lidx[1]].(fu[:,1]))), "county_2016_unemployment_original",                   0.0, 15.0, :yellowgreenblue, false);
plot_county(Dict(zip(fips,recover_funcs[lidx[2]].(fu[:,2]))), "county_2016_election_original",                      -1.0,  1.0, :redblue,          true);
plot_county(Dict(zip(fips,recover_funcs[lidx[1]].(fe[:,1]))), "county_2016_unemployment_expectation",                0.0, 15.0, :yellowgreenblue, false);
plot_county(Dict(zip(fips,recover_funcs[lidx[2]].(fe[:,2]))), "county_2016_election_expectation",                   -1.0,  1.0, :redblue,          true);

for (i,fs) in enumerate(fss)
plot_county(Dict(zip(fips,recover_funcs[lidx[1]].(fs[:,1]))), "county_2016_unemployment_sample"*@sprintf("%02d",i),  0.0, 15.0, :yellowgreenblue, false);
plot_county(Dict(zip(fips,recover_funcs[lidx[2]].(fs[:,2]))), "county_2016_election_sample"*@sprintf("%02d",i),     -1.0,  1.0, :redblue,          true);
end
