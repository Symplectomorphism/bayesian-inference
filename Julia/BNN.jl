#train bayesian NN with Gaussian approximation for the posterior

mutable struct Bayesian
    mymodel ::Function
    β       ::Float32
    α       ::Float32
    opt     ::ADAM
end

function Bayesian()
    β       = 0.00001f0
    α       = 1.0f0/3.0f0
    opt     = ADAM()
    Bayesian(nn_forward, β, α, opt)
end

# Turn a vector into a set of weights and biases
function unpack(nn_params::AbstractVector)
    W₁ = reshape(nn_params[1:2], 2, 1)
    b₁ = nn_params[3:4]

    # W₂ = reshape(nn_params[10:15], 2, 3)
    # b₂ = nn_params[16:17]

    Wₒ = reshape(nn_params[5:6], 1, 2)
    bₒ = nn_params[7:7]
    return W₁, b₁, Wₒ, bₒ
end

# Construct a neural network using Flux and return a predicted value.
function nn_forward(xs, nn_params::AbstractVector)
    W₁, b₁, Wₒ, bₒ = unpack(nn_params)
    nn = Chain(
        Dense(W₁, b₁, tanh),
        Dense(Wₒ, bₒ, σ)
    )
    return nn(xs)
end

function create_data(N)
    a0 = -0.3f0
    a1 = 0.5f0
    noise = 0.0f0*randn(Float32, 1, N)
    data_x = randn(Float32, 1, N)
    data_y = a0 .+ data_x .* a1 + noise

    return data_x, data_y
end

sig = sqrt(1.0 / b.α)
d = MvNormal(zeros(7), sig .* ones(7))
nn_params = rand(d, 1)[:]

w_test = rand(d,1)[:]
# This works for computing the Hessian at a given x (=1.0 here).
@info round.(Flux.hessian(w->b.mymodel([1.0], w)[1], w_test); digits=4)

loss(x, y) = b.β/2 * Flux.Losses.mse(b.mymodel(x, nn_params), y) + b.α/2 * sum(abs2, nn_params)

function train()
    train_x, train_y = create_data(10)

    dataset = Base.Iterators.repeated((train_x, train_y), 1)
    Flux.train!(loss, Flux.params(b.mymodel), dataset, b.opt)
end

# function plot_prediction(x) 
#     H = I       #compute hessian
#     grad = Flux.gradient( () -> 
#         BNN.mymodel([x])[1], Flux.params(BNN.mymodel))       #compute gradient with respect to weights
#     g = [] 
#     A = α*I + β*H
#     standev = 1.0f0/β + g'*(A\g)
#     Normal(BNN.mymodel(x), standev)
# end