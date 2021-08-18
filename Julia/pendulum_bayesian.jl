using ParameterizedFunctions, OrdinaryDiffEq, RecursiveArrayTools, Distributions
using DiffEqBayes
using Plots, StatsPlots

# # using CmdStan #required for using the Stan backend
# using DynamicHMC #required for DynamicHMC backend

# # f1 = @ode_def LotkaVolterra begin
# #     dx = a*x - x*y
# #     dy = -3*y + x*y
# # end a
   
# # p = [1.5]
# # u0 = [1.0,1.0]
# # tspan = (0.0,10.0)
# # prob1 = ODEProblem(f1,u0,tspan,p)

# # σ = 0.01                         # noise, fixed for now
# # t = collect(1.:10.)   # observation times
# # sol = solve(prob1,Tsit5())
# # priors = [Normal(1.5, 1)]
# # randomized = VectorOfArray([(sol(t[i]) + σ * randn(2)) for i in 1:length(t)])
# # data = convert(Array,randomized)

# f1 = @ode_def begin
#     dx = a*x - b*x*y
#     dy = -c*y + d*x*y
# end a b c d

# p = [1.5,1.0,3.0,1.0]
# u0 = [1.0,1.0]
# tspan = (0.0,10.0)
# prob1 = ODEProblem(f1,u0,tspan,p)

# sol = solve(prob1,Tsit5())
# σ = 0.01  
# t = collect(range(1,stop=10,length=10))
# randomized = VectorOfArray([(sol(t[i]) + σ * randn(2)) for i in 1:length(t)])
# data = convert(Array,randomized)


# priors = [truncated(Normal(1.5,0.1),0,2),truncated(Normal(1.0,0.1),0,1.5),
#           truncated(Normal(3.0,0.1),0,4),truncated(Normal(1.0,0.1),0,2)]

# # bayesian_result = stan_inference(prob1,t,data,priors;
# #                                  num_samples=100,num_warmup=500,
# #                                  vars = (StanODEData(),InverseGamma(4,1)))

# bayesian_result = turing_inference(prob1,Tsit5(),t,data,priors)


# # theta1 = bayesian_result.chain_results[:,["theta.1"],:]
# # theta2 = bayesian_result.chain_results[:,["theta.2"],:]
# # theta3 = bayesian_result.chain_results[:,["theta.3"],:]
# # theta4 = bayesian_result.chain_results[:,["theta.4"],:]

# # mean(theta1.value[:,:,1])



f2 = @ode_def begin
    dx = y
    dy = sin(x) - a*sin(x) - b*y
end a b

p = [2.0, 1.0]
u0 = [0.1,0.0]
tspan = (0.0,10.0)
prob2 = ODEProblem(f2,u0,tspan,p)

sol = solve(prob2,Tsit5())
# σ = 0.01  
t = collect(range(1,stop=tspan[2],length=100))
# randomized = VectorOfArray([(sol(t[i]) + 0.01 * randn(2)) for i in 1:length(t)])
randomized = VectorOfArray([(zeros(2) + 0.01 * randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)
# data = zeros(2, 10)

priors = [truncated(Normal(0.1,0.1),0,3),truncated(Normal(0.0,0.1),0,2)]

bayesian_result = turing_inference(prob2,Tsit5(),t,data,priors)

plot(bayesian_result)