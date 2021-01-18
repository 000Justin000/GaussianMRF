using Plots;
using URIParser;
using VegaLite;
using VegaDatasets;

function plot_ising(coords, labels, L, side_length=500, network_name="tmp")
    if ndims(coords) == 1
        coords = hcat(coords...);
    end

    h = plot(framestyle=:box, xlim=(-1.1, 1.1),
                              ylim=(-1.1, 1.1),
                              xticks=[],
                              yticks=[],
                              legend=:right, size=(side_length,side_length));

    scatter!(h, coords[1,:], coords[2,:], markersize=5,
                                          markeralpha=[i in L ? 1.00 : 0.05 for i in 1:length(coords)],
                                          markercolor=[label >= 0 ? :red : :blue for label in labels],
                                          markerstrokewidth = 0.1,
                                          label="");

    savefig(h, network_name*".svg");
end

function plot_county(dat, fname="tmp", min_v=-1.0, max_v=1.0, scheme=:redblue, reverse=false)
    us10m = dataset("us-10m");
    df = DataFrame(Dict("id"=>collect(keys(dat)), "value"=>(x->min(max_v,max(min_v,x))).(collect(values(dat)))));

    h = @vlplot(
            width=1500, height=1100,
            mark={:geoshape, stroke=:gray, strokeWidth=0.1},
            data={values=us10m, format={type=:topojson, feature=:counties}},
            transform=[{lookup=:id, from={data=df, key=:id, fields=["value"]}}],
            projection={type=:albersUsa},
            color={"value:q", scale={domain=[min_v,max_v], scheme=scheme, reverse=reverse}, legend=nothing},
            config={view={stroke=nothing},axis={grid=false}}
        );

    save("figs/"*fname*".svg", h);
    save("figs/"*fname*".png", h);
end
