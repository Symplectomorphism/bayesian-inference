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
    dy = sin(x) - a*y*( 1/2*y^2 + (-1+cos(x)) )
    dz = -a*y^2*z^2
end a


function energy_pendulum(du, u, p, t)
    H = 1/2*u[2]^2 + (1+cos(u[1]))
    Href = 2.0
    H̃ = H - Href
    du[1] = u[2]
    du[2] = sin(u[1]) - 10.0*u[2]*H̃
end

u0 = [π+10*π/180,0.0]
tspan = (0.0, 10.0)
prob = ODEProblem(energy_pendulum, u0, tspan)
sol = solve(prob,Tsit5())
plot(sol,vars=(1,2))


p = [-10.0]
u0 = [π+10*π/180, 0.0, 1/2*0.0^2 + (1+cos( π+10*π/180 ))]
tspan = (0.0,10.0)
prob2 = ODEProblem(f2,u0,tspan,p)

sol = solve(prob2,Tsit5(),save_idxs=[3])
# σ = 0.01  
t = collect(range(1,stop=tspan[2],length=100))
# randomized = VectorOfArray([(sol(t[i]) + 0.01 * randn(2)) for i in 1:length(t)])
randomized = VectorOfArray([(zeros(3) + 0.01 * randn(3)) for i in 1:length(t)])
data = convert(Array,randomized)
# data = zeros(2, 10)

# priors = [truncated(Normal(0.0,0.1),0,3),truncated(Normal(0.0,0.1),0,2)]
# priors = [Normal(-10.0, 100.0)]
priors = [truncated(Normal(-10.0,5.0),-10,5)]

bayesian_result = turing_inference(prob2,Tsit5(),t,data,priors)

plot(bayesian_result)