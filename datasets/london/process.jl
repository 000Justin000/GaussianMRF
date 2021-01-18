using CSV;
using Plots;

code = CSV.read("code.csv", header=1);
W = CSV.read("W.csv", header=1)
dat = CSV.read("MaycoordLondon.csv", header=1);

info = join(code[:,2:end], dat, on=names(code)[2] => names(dat)[1], kind=:left);

indices = (!ismissing).(info[:,2]);
W0 = Matrix(W[:,2:end] .!== 0.0)[indices, indices];
info0 = info[indices,:];

h = plot(size=(550,500));

for i in 1:size(W0,1)
    for j in i:size(W0,2)
        W0[i,j] && plot!(h, [info0[i,names(info0)[8]], info0[j,names(info0)[8]]], [info0[i,names(info0)[9]], info0[j,names(info0)[9]]], color=1, label="");
    end
end

scatter!(h, info0[:,names(info)[8]], info0[:,names(info)[9]], label="");
display(h);
