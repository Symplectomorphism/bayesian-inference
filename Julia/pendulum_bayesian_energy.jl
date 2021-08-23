using ParameterizedFunctions, OrdinaryDiffEq, RecursiveArrayTools, Distributions
using DiffEqBayes, Turing
using Plots, StatsPlots

Turing.setprogress!(true)

f2 = @ode_def begin
    dx = y
    dy = sin(x) + 
            4*min(abs(z), 0.25) * (-a*y*( 1/2*y^2 + (-1+cos(x)) )) + 
            4*max(0.25, abs(z))*(-b*sin(x) - c*y)
    dz = -a*y^2*z
end a b c


# f2 = @ode_def begin
#     dx = y
#     dy = sin(x) + 
#             (-a*y*( 1/2*y^2 + (-1+cos(x)) ))
#     dz = -a*y^2*z
# end a


p = [0.0, 0.0, 0.0]
u0 = [π+10*π/180, 0.0, 1/2*0.0^2 + (-1+cos( π+10*π/180 ))]
tspan = (0.0,50.0)
prob2 = ODEProblem(f2,u0,tspan,p)

sol = solve(prob2,Tsit5(),save_idxs=[1,2])
# σ = 0.01  
t = collect(range(20,stop=tspan[2],length=30))
# randomized = VectorOfArray([(sol(t[i]) + 0.01 * randn(2)) for i in 1:length(t)])
randomized = VectorOfArray([(zeros(2) + 0.001 * randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)
data = zeros(2, length(t))

# priors = [truncated(Normal(0.0,0.1),0,3),truncated(Normal(0.0,0.1),0,2)]
# priors = [Normal(-10.0, 100.0)]
# priors = [truncated(Normal(-10.0,5.0),0,5)]
priors = [truncated(Normal(0.0,5.0),0,5), 
          truncated(Normal(2.0,1.0),0,4),
          truncated(Normal(1.0,1.0),0,0.1)]

bayesian_result = turing_inference(prob2,Tsit5(),t,data,priors;
                    save_idxs=[1,2],
                    sampler = Turing.NUTS(1_000, 0.65))

plot(bayesian_result)

display( findmax(bayesian_result.value.data, dims=1)[1][1:3] )


function energy_pendulum(du, u, p, t)|
    H = 1/2*u[2]^2 + (1+cos(u[1]))
    Href = 2.0
    H̃ = H - Href
    du[1] = u[2]
    du[2] = 4*min(abs(u[3]), 0.25)*(-p[1]*u[2]*H̃) + sin(u[1]) + 
                + 4*max(0.25, abs(u[3])) * (-p[2]*sin(u[1]) - p[3]*u[2])
    du[3] = -p[1]*u[2]^2*u[3]
end

u0 = [π+10*π/180, 0.0, 1/2*0.0^2 + (-1+cos( π+10*π/180 ))]
tspan = (0.0, 60.0)
prob = ODEProblem(energy_pendulum, u0, tspan, 
    findmax(bayesian_result.value.data, dims=1)[1][1:3])
sol = solve(prob,Tsit5())
plot(sol,vars=(1,2))
plot(sol,vars=(3))