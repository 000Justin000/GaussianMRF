using Juno;
using Statistics;
using Random;
using Printf;
using LightGraphs;
using LinearAlgebra;
using SparseArrays;
using MLBase;
using CatViews;
using Optim;
import Flux: train!;

eye(n) = diagm(0=>ones(n));
speye(n) = spdiagm(0=>ones(n));
A2D(A) = spdiagm(0=>sum(A,dims=1)[:]);

function normalized_laplacian(G)
    @assert sum(diag(adjacency_matrix(G))) == 0;

    # if a vertex has degree 0, then the corresponding row and col are empty
    L = spdiagm(0=>degree(G).^-0.5) * laplacian_matrix(G) * spdiagm(0=>degree(G).^-0.5);

    @assert !any(isnan.(L));

    return L;
end

function rand_split(input, pctgs; seed_val=0)
    """
    Args:
     input: total number of data points, or data points themselves
     pctgs: percentages of different splits

    Returns:
         indices for different splits
    """
    @assert abs(sum(pctgs)-1.0) < 1.0e-10;

    Random.seed!(seed_val);

    if typeof(input) == Int
        n = input;
        shuffled_data = randperm(n);
    elseif typeof(input) <: AbstractVector
        n = length(input);
        shuffled_data = input[randperm(n)];
    else
        error("unexpected input argument");
    end

    pr(pctg) = Int64(ceil(pctg*n));
    fcs = cumsum(vcat(0, pctgs));

    return tuple([shuffled_data[pr(fcs[i])+1:pr(fcs[i+1])] for i in 1:length(pctgs)]...);
end

function array2str(arr)
    """
    Args:
       arr: array of data
       fmt: format string
    Return:
       string representation of the array
    """

    (typeof(arr[1]) <: String) || (arr = map(x->@sprintf("%10.5f", x), arr));
    return join(arr, ", ");
end

function R2(y_, y)
    """
    Args:
        y_: predicted labels
         y: true labels

    Return:
        coefficients of determination
    """
    @assert ((ndims(y_) == ndims(y) == 1) || (size(y_,1) == size(y,1) == 1)) "unexpected input size"

    # println(sum((y_[:] .- y[:]).^2.0))
    # println(sum((y[:] .- mean(y[:])).^2.0))

    return 1.0 - sum((y_[:] .- y[:]).^2.0) / sum((y[:] .- mean(y[:])).^2.0);
end

function probmax(y_, y)
    """
    Args:
        y_: predicted probabilities
         y: true labels in one-hot encoding

    Return:
        accuracy
    """
    @assert (ndims(y_) == ndims(y) == 2) "unexpected input size"

    return sum(y[argmax(y_; dims=1)]) / size(y,2);
end

function expansion(m, ids)
    """
    Args:
         m: overall dimension
       ids: a length m_ vector with indices indicating location

    Returns:
         Ψ: a m x m_ matrix that expand a vector of dimension m_ to a vector of dimension m
    """

    m_ = length(ids);

    II = Vector{Int}();
    JJ = Vector{Int}();
    VV = Vector{Float64}();

    for (i,id) in enumerate(ids)
        push!(II, id);
        push!(JJ, i);
        push!(VV, 1.0);
    end

    return sparse(II, JJ, VV, m, m_);
end

function reset_grad!(xs...)
    for x in xs
        x.grad .= 0;
    end
end

function train!(loss, θs::Vector, mini_batches::Vector, opts::Vector; start_opts=zeros(Int,length(opts)), cb=()->(), cb_skip=1)
    """
    extend training method to allow using different optimizers for different parameters
    """

    ps = Params(vcat(collect.(θs)...));
    for (i,mini_batch) in enumerate(mini_batches)
        gs = gradient(ps) do
            loss(mini_batch...);
        end

        for (θ,opt,start_opt) in zip(θs,opts,start_opts)
            (i > start_opt) && update!(opt, θ, gs);
        end

        (i % cb_skip == 0) && cb();
    end
end
