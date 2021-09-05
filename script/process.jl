using CSV;
using DataFrames;
using Printf;
using Statistics;

function read_transductive(df, split_ratio, info)
    return parse.(Float64, strip.(split(filter(row->(row[:Column1]==split_ratio) && (strip(row[:Column2])==info), df)[1,:Column3], ',')));
end

# fname = "transductive_county_facebook_2016_income";
# fname = "transductive_county_facebook_2016_education";
# fname = "transductive_county_facebook_2016_unemployment";
# fname = "transductive_county_facebook_2016_election";
# fname = "transductive_climate_2008_airT";
# fname = "transductive_climate_2008_landT";
# fname = "transductive_climate_2008_precipitation";
# fname = "transductive_climate_2008_sunlight";
# fname = "transductive_climate_2008_pm2.5";
# fname = "transductive_ward_2016_income";
# fname = "transductive_ward_2016_edu";
# fname = "transductive_ward_2016_age";
# fname = "transductive_ward_2016_election";
# fname = "transductive_facebook_sh050m"
# fname = "transductive_facebook_sh100m"
# fname = "transductive_facebook_sh500m"
# fname = "transductive_twitch_PTBR_true_4"

# df = CSV.read("../results/"*fname, DataFrame; datarow=1, header=0, delim=';');
# acs = ["ac_lp", "ac_lr", "ac_lgc", "ac_sgc", "ac_gcn", "ac_lgcrp", "ac_sgcrp", "ac_gcnrp"];
# for ac in acs
#     @printf("%5.2f ", mean(read_transductive(df, 0.3, ac)));
# end
# @printf("\n");
 
# Kxs = ["Kx_sgc", "Kx_gcn"];
# for Kx in Kxs
#     @printf("%5.2f ", mean(read_transductive(df, 0.3, Kx)));
# end
# @printf("\n");
 
# αxs = ["αx_lp", "αx_lgc", "αx_lgcrp", "αx_sgcrp", "αx_gcnrp"];
# for αx in αxs
#     @printf("%5.2f ", mean(read_transductive(df, 0.3, αx)));
# end
# @printf("\n");



# Here the seeds are not fixed
# fname = "transductive_synthetic_WattsStrogatzOriginal_+0.0_1"
# fname = "transductive_synthetic_WattsStrogatzOriginal_+1.0_1"
# fname = "transductive_synthetic_WattsStrogatzOriginal_+2.0_1"

# Fix random seed to be 0 during data generation
# fname = "transductive_synthetic_WattsStrogatzOriginal_+0.0_1"
# fname = "transductive_synthetic_WattsStrogatzOriginal_+1.0_1"
# fname = "transductive_synthetic_WattsStrogatzOriginal_+2.0_1"
# fname = "transductive_synthetic_WattsStrogatz_+0.0_1"
# fname = "transductive_synthetic_WattsStrogatz_+1.0_1"
# fname = "transductive_synthetic_WattsStrogatz_+2.0_1"
# fname = "transductive_synthetic_StochasticBlockModel_+0.0_1"
# fname = "transductive_synthetic_StochasticBlockModel_+1.0_1"
# fname = "transductive_synthetic_StochasticBlockModel_+2.0_1"
# fname = "transductive_synthetic_DegreeCorrectedStochasticBlockModel_+0.0_1"
# fname = "transductive_synthetic_DegreeCorrectedStochasticBlockModel_+1.0_1"
  fname = "transductive_synthetic_DegreeCorrectedStochasticBlockModel_+2.0_1"
# fname = "transductive_synthetic_BarabasiAlbert_+0.0_1"
# fname = "transductive_synthetic_BarabasiAlbert_+1.0_1"
# fname = "transductive_synthetic_BarabasiAlbert_+2.0_1"

  acs = ["ac_lp", "ac_lr", "ac_lgc", "ac_sgc", "ac_gcn", "ac_lgcrp", "ac_sgcrp", "ac_gcnrp"];
  for ac in acs
      mean_values = [];
      for i in 1:5
          df = CSV.read("../results/"*fname*"_"*string(i), DataFrame; datarow=1, header=0, delim=';')
          append!(mean_values, read_transductive(df, 0.3, ac));
      end
      @printf("%5.2f ", mean(mean_values));
  end
  @printf("\n");

  Kxs = ["Kx_sgc", "Kx_gcn"];
  for Kx in Kxs
      mean_values = [];
      for i in 1:5
          df = CSV.read("../results/"*fname*"_"*string(i), DataFrame; datarow=1, header=0, delim=';')
          append!(mean_values, read_transductive(df, 0.3, Kx));
      end
      @printf("%5.2f ", mean(mean_values));
  end
  @printf("\n");

  αxs = ["αx_lp", "αx_lgc", "αx_lgcrp", "αx_sgcrp", "αx_gcnrp"];
  for αx in αxs
      mean_values = [];
      for i in 1:5
          df = CSV.read("../results/"*fname*"_"*string(i), DataFrame; datarow=4, header=0, delim=';')
          append!(mean_values, read_transductive(df, 0.3, αx));
      end
      @printf("%5.2f ", mean(mean_values));
  end
  @printf("\n");
