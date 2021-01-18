using Random;
using StatsBase;
using LightGraphs;
using LinearAlgebra
using SparseArrays;
using Arpack;
using Flux;
using Flux.Tracker: data, track, @grad, forward, Params, update!, back!, grad, gradient;
using Flux.Tracker: TrackedReal, TrackedVector, TrackedMatrix;
using IterativeSolvers;
using Printf;
using PyCall;

include("utils.jl");

function mBCG(mmm_A::Function, B::Array{Float64,2}; k::Int=size(B,1), tol=1.0e-12)
    """
    Args:
     mmm_A: matrix matrix multiplication routine
         B: right-hand-side vectors
         k: max # of iterations
       tol: error tolerance

    Returns:
         X: solution vectors
        TT: Lanczos tridiagonal matrices
    """

    n,t = size(B);
    X = zeros(n,t);
    R = B - mmm_A(X);
    P = R;
    α = zeros(t);
    β = zeros(t);
    γ = sum(R.*R, dims=1)[:];

    T = [(dv=Vector{Float64}(), ev=Vector{Float64}()) for _ in 1:t];

    tol_vec = tol .+ tol*sqrt.(γ);
    for j in 1:k
        if all(sqrt.(γ) .< tol_vec)
            break;
        end

        AP = mmm_A(P);
        α_ =  γ ./ sum(P.*AP, dims=1)[:];
        X +=  P .* α_';
        R -= AP .* α_';
        γ_ = sum(R.*R, dims=1)[:];
        β_ = γ_ ./ γ;
        P  = R + P .* β_';

        for i in 1:t
            if j == 1
                push!(T[i].dv, 1.0/α_[i]);
            else
                push!(T[i].dv, 1.0/α_[i]+β[i]/α[i]);
                push!(T[i].ev, sqrt(β[i])/α[i]);
            end
        end

        # record current values for next iteration
        α = α_;
        β = β_;
        γ = γ_;
    end

    return X, [SymTridiagonal(dv,ev) for (dv,ev) in T];
end

function parallel_mBCG(mmm_A::Function, B::Array{Float64,2}; k::Int=size(B,1), tol=1.0e-6)
    t = size(B,2);
    d = Threads.nthreads();
    b = Int(ceil(t/d));

    wl = [(i-1)*b+1:min(i*b,t) for i in 1:d];
    Xs = Vector{Array{Float64,2}}(undef,d);
    TTs = Vector{Vector{SymTridiagonal{Float64,Vector{Float64}}}}(undef,d);

    Threads.@threads for i in 1:d
        Xs[i], TTs[i] = mBCG(mmm_A, B[:,wl[i]]; k=k, tol=tol);
    end

    return hcat(Xs...), vcat(TTs...);
end

#-------------------------------------
# model dependent part
#-------------------------------------
function getdiagΓ(α; A)
    return sum(α_*collect(diag(A_)) for (α_,A_) in zip(α,A));
end

function getΓ(α; A)
    return sum(α_*A_ for (α_,A_) in zip(α,A));
end

function get∂Γ∂α(α::Vector{Float64}; A)
    return A;
end
#-------------------------------------

logdetΓ(α::TrackedVector; A, P, t, k) = track(logdetΓ, α; A=A, P=P, t=t, k=k);
@grad function logdetΓ(α; A, P, t, k)
    """
    Args:
         α: model parameter vector
         A: matrix vector
         P: index set
         t: # of trial vectors
         k: # of Lanczos tridiagonal iterations

    Return:
         log determinant of the principle submatrix ΓPP
    """

    (length(P) == 0) && return 0.0, Δ -> (zeros(length(α)), 0.0);

    α = data(α);

    n = length(P);
    Z = randn(n,t);

    Γ = getΓ(α; A=A);
    ∂Γ∂α = get∂Γ∂α(α; A=A);

    X, TT = mBCG(Y->Γ[P,P]*Y, Z; k=k);

    vv = 0;
    for T in TT
        eigvals, eigvecs = eigen(T);
        vv += sum(eigvecs[1,:].^2 .* log.(eigvals));
    end
    Ω = vv*n/t;

    trΓiM(M) = sum(X.*(M[P,P]*Z))/t;
    ∂Ω∂α = map(trΓiM, ∂Γ∂α);

    return Ω, Δ -> tuple(Δ*∂Ω∂α);
end

function test_logdetΓ(n=100)
    G = random_regular_graph(n, 3);
    A = [laplacian_matrix(G), spdiagm(0=>rand(n))];
    L = randperm(n)[1:div(n,2)];

    #------------------------
    p = param(rand(2));
    getα() = p[:];
    #------------------------

    #------------------------
    # true value
    #------------------------
    Γ = Array{eltype(p)}(undef, n, n);
    Γ .= getΓ(getα(); A=A);
    Ω = logdet(Γ[L,L]);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       %8.3f    [%s]\n", data(Ω), array2str(Tracker.grad(p)));
    reset_grad!(p);
    #------------------------

    #------------------------
    # approximation
    #------------------------
    Ω = logdetΓ(getα(); A=A, P=L, t=128, k=128);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("approximate:    %8.3f    [%s]\n", data(Ω), array2str(Tracker.grad(p)));
    reset_grad!(p);
    #------------------------
end

quadformSC(α::TrackedVector, rL::TrackedVector; A, L) = track(quadformSC, α, rL; A=A, L=L);
@grad function quadformSC(α, rL; A, L)
    """
    Args:
         α: model parameter vector
        rL: noise on vertex set L
         A: matrix vector
         L: index set

    Return:
         quadratic form: rL' (ΓLL - ΓLU ΓUU^-1 ΓUL) rL
    """

    α = data(α);
    rL = data(rL);

    Γ = getΓ(α; A=A);
    ∂Γ∂α = get∂Γ∂α(α; A=A);

    U = setdiff(1:size(A[1],1), L);

    Ω = rL'*Γ[L,L]*rL - rL'*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    quadform_partials(M) = rL'*M[L,L]*rL - rL'*M[L,U]*cg(Γ[U,U],Γ[U,L]*rL) + rL'*Γ[L,U]*cg(Γ[U,U],M[U,U]*cg(Γ[U,U],Γ[U,L]*rL)) - rL'*Γ[L,U]*cg(Γ[U,U],M[U,L]*rL);
    ∂Ω∂α = map(quadform_partials, ∂Γ∂α);
    ∂Ω∂rL = 2*Γ[L,L]*rL - 2*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    return Ω, Δ -> tuple(Δ*∂Ω∂α, Δ*∂Ω∂rL);
end

function test_quadformSC(n=100)
    G = random_regular_graph(n, 3);
    A = [laplacian_matrix(G), speye(n)];

    #------------------------
    L = randperm(n)[1:div(n,2)];
    U = setdiff(1:n, L);
    rL = param(randn(div(n,2)));
    getrL() = rL[:];
    #------------------------

    #------------------------
    p = param(rand(2));
    getα() = p[:];
    #------------------------

    #------------------------
    # true value
    #------------------------
    Γ = Array{eltype(p)}(undef, n, n);
    Γ .= getΓ(getα(); A=A);
    Γ = Tracker.collect(Γ);
    SC = Γ[L,L] - Γ[L,U]*inv(Γ[U,U])*Γ[U,L];
    Ω = getrL()' * SC * getrL();
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(rL)[1:10]));
    reset_grad!(p, rL);
    #------------------------

    #------------------------
    # approximation
    #------------------------
    Ω = quadformSC(getα(), getrL(); A=A, L=L);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(rL)[1:10]));
    reset_grad!(p, rL);
    #------------------------
end

ΓX(α::TrackedVector, X; A, U, L) = track(ΓX, α, X; A=A, U=U, L=L);
@grad function ΓX(α, X; A, U, L)
    """
    Args:
         α: model parameter vector
         X: matrix
         A: matrix vector
         U: index set
         L: index set

    Return:
         Y = ΓUL X
    """
    @assert (size(X,1) == length(L))

    α = data(α);
    X = data(X);

    Γ = getΓ(α; A=A);
    ∂Γ∂α = get∂Γ∂α(α; A=A);

    function sensitivity(ΔY)
        ΔΓUL = ΔY * X';
        ΔX = Γ[U,L]' * ΔY;

        return tuple([sum(ΔΓUL .* ∂Γ∂α_[U,L]) for ∂Γ∂α_ in ∂Γ∂α], ΔX);
    end

    return Γ[U,L]*X, sensitivity;
end

function test_ΓB(n=100, m=20)
    G = random_regular_graph(n, 3);
    A = [adjacency_matrix(G), speye(n)];

    #------------------------
    L = randperm(n)[1:div(n,2)];
    U = setdiff(1:n, L);
    X = param(randn(length(L), m));
    C = randn(length(U), m);
    #------------------------

    #------------------------
    p = param(rand(2));
    getα() = p[:];
    #------------------------

    #------------------------
    # true value
    #------------------------
    Γ = getΓ(getα(); A=A);
    Ω = sum((Γ[U,L]*X) .* C);
    #------------------------
    Tracker.back!(Ω, 1);
    ΔX0 = Tracker.grad(X);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(X)[:][1:10]));
    reset_grad!(p, X);
    #------------------------

    #------------------------
    # approximation
    #------------------------
    Ω = sum(ΓX(getα(), X; A=A, L=L, U=U) .* C);
    #------------------------
    Tracker.back!(Ω, 1);
    ΔX1 = Tracker.grad(X);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(X)[:][1:10]));
    reset_grad!(p, X);
    #------------------------

    @assert all(ΔX0 .== ΔX1);
end

chol(A::TrackedArray) = track(chol, A);
@grad function chol(A)
    A = data(A);
    CF = cholesky(A);
    L, U = CF.L, CF.U;

    Φ(A) = LowerTriangular(A) - 0.5 * Diagonal(A);

    function sensitivity(ΔL)
        S = inv(U) * Φ(U * LowerTriangular(ΔL)) * inv(L);
        # return tuple(Matrix(S + S' - Diagonal(S)));
        return tuple(Matrix(0.5 * (S + S')));
    end

    return Matrix(L), sensitivity;
end

function test_chol(n=10)
    X = randn(n,n);
    A0 = X * X';
    A = param(A0);
    b = randn(n);

    Ω(A, b) = b' * A * b;

    reset_grad!(A);
    Tracker.back!(Ω(chol(A), b), 1);
    @printf("reverse-mode automatic differentiation\n");
    display(A.grad);
    @printf("\n\n");

    B = Matrix{eltype(A)}(undef, n, n);
    B .= A;
    reset_grad!(A);
    Tracker.back!(Ω(cholesky(B).L, b), 1);
    @printf("elementwise automatic differentation\n");
    display(A.grad);
    @printf("\n\n");

    ϵ = 1.0e-6;
    sen = zeros(n,n);
    for i in 1:n
        for j in 1:n
            Ap, Am = Array(A0), Array(A0);
            Ap[i,j] += ϵ;
            Ap[j,i] += ϵ;
            Am[i,j] -= ϵ    ;
            Am[j,i] -= ϵ;

            sen[i,j] = (Ω(cholesky(Ap).L, b) - Ω(cholesky(Am).L, b)) / (4 * ϵ);
            sen[j,i] = (Ω(cholesky(Ap).L, b) - Ω(cholesky(Am).L, b)) / (4 * ϵ);
        end
    end

    @printf("finite-difference\n");
    display(sen);
end

function trinv(Γ; P, t, k)
    """
    Args:
         Γ: input matrix
         P: index set
         t: # of trial vectors
         k: # of Lanczos tridiagonal iterations

    Return:
         trace of the principle submatrix of the inverse tr(Γ^{-1}[P,P])
    """

    (length(P) == 0) && return 0.0;

    n = size(Γ,1);
    Z = zeros(n,t);
    Z[P,:] = randn(length(P),t);

    X, _ = mBCG(Y->Γ*Y, Z; k=k);

    Ω = sum(X.*Z)/t;

    return Ω;
end
