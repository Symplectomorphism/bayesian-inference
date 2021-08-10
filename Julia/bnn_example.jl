# Import libraries
using Turing, Flux, Plots, Random, ReverseDiff
# pyplot()
gr()

# Hide or show sampling progress.
Turing.setprogress!(true)

# Use reverse_diff due to the number of parameters in neural networks
Turing.setadbackend(:reversediff)


"""
Our goal here is to use a Bayesian neural network to classify points in an 
artificial dataset. The code below generates data points arranged in a box-like 
pattern and displays a graph of the dataset we'll be working with.
"""

# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5
append!(xt1s, Array([[x1s[i] - 5; x2s[i] + 5] for i = 1:M]))

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

# Store all the data for later
xs = [xt1s; xt0s]
ts = [ones(2*M); zeros(2*M)]

# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1, y1, color="red", clim=(0,1))
    Plots.scatter!(x2, y2, color="blue", clim=(0,1))
end

plot_data()


"""
The next step is to define a feedforward neural network where we express our
parameters as distributions, and not single points as traditional neural
networks. The two functions below, `unpack` and `nn_forward` are helper
functions we need when we specify our model in Turing.

`unpack` takes a vector of parameters and partitions them between weights and
biases. 

`nn_forward` constructs a neural network with the variables generated in
`unpack` and returns a prediction based on the weights provided.

The `unpack` and `nn_forward` functions are explicitly designed to create a
neural network with two hidden layers and one output layer.
"""

# Turn a vector into a set of weights and biases
function unpack(nn_params::AbstractVector)
    W₁ = reshape(nn_params[1:6], 3, 2)
    b₁ = nn_params[7:9]

    W₂ = reshape(nn_params[10:15], 2, 3)
    b₂ = nn_params[16:17]

    Wₒ = reshape(nn_params[18:19], 1, 2)
    bₒ = nn_params[20:20]
    return W₁, b₁, W₂, b₂, Wₒ, bₒ
end

# Construct a neural network using Flux and return a predicted value.
function nn_forward(xs, nn_params::AbstractVector)
    W₁, b₁, W₂, b₂, Wₒ, bₒ = unpack(nn_params)
    nn = Chain(
        Dense(W₁, b₁, tanh),
        Dense(W₂, b₂, tanh),
        Dense(Wₒ, bₒ, σ)
    )
    return nn(xs)
end


"""
The probabilistic model specification below creates a `params` variable, which
has 20 normally distributed variables. Each entry in the `params` vector
represents weights and biases of our neural net.
"""

# Create a regularization term and a Gaussian prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)

# Specify the probabilistic model
@model function bayes_nn(xs, ts)
    # Create the weight and bias vector
    nn_params ~ MvNormal(zeros(20), sig .* ones(20))

    # Calculate predictions for the inputs given the weights and biases in theta.
    preds = nn_forward(xs, nn_params)

    # Observe each prediction.
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end


"""
Inference can now be performed by calling `sample`. We use the `HMC` sampler
here.
"""

# Perform inference.
N = 5000
ch = sample(bayes_nn(hcat(xs...), ts), HMC(0.05, 4), N)

"""
Now we extract the weights and biases from the sampled chain. We'll use these
primarily in determining how good a classifier our model is.
"""

# Extract all weight and bias parameters.
theta = MCMCChains.group(ch, :nn_params).value