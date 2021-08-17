using Turing, Flux
using ReverseDiff
using Random, LinearAlgebra
using StatsPlots

Turing.setprogress!(true)
Turing.setadbackend(:reversediff)

# Turn a vector into a set of weights and biases.
function unpack(nn_params::AbstractVector)
    W₁ = reshape(nn_params[1:2], 2, 1)
    b₁ = nn_params[3:4]

    Wₒ = reshape(nn_params[5:6], 1, 2)
    bₒ = nn_params[7:7]
    return W₁, b₁, Wₒ, bₒ
end

# Construct a neural network using Flux and return a predicted value.
function nn_forward(xs, nn_params::AbstractVector)
    W₁, b₁, Wₒ, bₒ  = unpack(nn_params)
    nn              = Chain(
                        Dense(W₁, b₁),
                        Dense(Wₒ, bₒ)
                    )
    return nn(xs)
end

function create_data(data_x, noise_standev)
    a0      = -0.3f0
    a1      = 0.5f0
    noise   = noise_standev*randn()
    data_y  = a0 .+ data_x .* a1 + noise

    return data_y
end

# Create a regularization term and a Gaussain prior variance term.
alpha = 1.0/10.0
sig = sqrt(1.0 / alpha)
β = 1.0

# Specify the probabilistic model.
@model function bayes_nn(xs, ts)
    M = 7
    nn_params = Vector(undef, M)
    # Create the weight and bias vector.
    for i = 1:M
        nn_params[i] ~ Normal(0, sig)
    end
    
    # Calculate predictions for the inputs given the weights
    # and biases in theta.
    preds = nn_forward(xs, nn_params)
    
    N = length(xs)
    # Observe each prediction.
    for i = 1:N
        ts[i] ~ Normal(preds[i], 1/√β)
    end
end;

# Perform inference.
N_data = 20
xs  = randn(N_data)
# ts = hcat([create_data(xs[i], 1.0/sqrt(β)) for i = 1:length(xs)])
ts = hcat([create_data(xs[i], 1.0/sqrt(10_000)) for i = 1:length(xs)])
ch = sample(bayes_nn(hcat(xs...), ts), HMC(0.05, 10), 1_000);
# ch = sample(bayes_nn(hcat(xs...), ts), NUTS(1_000, 0.65), 10_000);

theta = MCMCChains.group(ch, :nn_params).value;
_, i = findmax(ch[:lp])
i = i.I[1]

W₁, b₁, Wₒ, bₒ = unpack(theta[i,:])
println("a0_pred = $((Wₒ*b₁ + bₒ)[1]), a1_pred = $((Wₒ*W₁)[1])")
println("a0_real = -0.3, a1_pred = 0.5")

StatsPlots.plot(ch)
#TODO: turn the trained parameters into a vector and make predictions

# Return the average predicted value across
# multiple weights.
function nn_predict(x, theta, num)
    mean([nn_forward(x, theta[i,:])[1] for i in 1:10:num])
end;