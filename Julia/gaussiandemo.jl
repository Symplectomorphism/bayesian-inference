# Import packages
using Turing
using StatsPlots

# Define a simple Normal model with unknown mean
@model function gdemo(x, y)
    s² = 3
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
end

c = sample(gdemo(1.5, 2), NUTS(0.65), 10_000)