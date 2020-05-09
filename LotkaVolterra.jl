using Plots; gr()
using DifferentialEquations
#--- Lotka-Volterra equation
#  predator–prey equations
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

#--- Lotka-Volterra parameters
Base.@kwdef struct LV_Param
  u0::Vector = [1.0, 1.0] # initial condition
  p::Vector = [1.5, 1.0, 3.0, 1.0]# α, β, δ, γ
  tspan::Tuple = (0.0, 10.0) # time span of simulation
  save_freq::Float32 = 0.1
end

#--- solve ODE

function solve_lotka_volterra(p)
  prob = ODEProblem(lotka_volterra,p.u0,p.tspan,p.p)
  sol = solve(prob,Tsit5(),saveat=p.save_freq)
  return sol
end

#--- spiral
Base.@kwdef struct Spiral_Param
  u0::Vector = [2.0, 0.0] # initial condition
  p::Vector = [-0.1f0, 2.0f0, -2.0f0, -0.1f0]# α, β, δ, γ
  tspan::Tuple = (0.0, 10.0) # time span of simulation
  save_freq::Float32 = 0.1
end

function spiral(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x^3 + β*y^3
  du[2] = dy = δ*x^3 + γ*y^3
end

function solve_spiral(p)
  prob = ODEProblem(spiral,p.u0,p.tspan,p.p)
  sol = solve(prob,Tsit5(),saveat=p.save_freq)
  return sol
end
