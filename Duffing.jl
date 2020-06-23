
using Plots; gr()
using DifferentialEquations
#--- Duffing equation
function duffing(du,u,p,t)
  x, y = u
  α, β, δ, γ, ω = p
  du[1] = dx = y
  du[2] = dy = -δ*y - β*x^3 - α*x + γ*cos(ω*t)
end

#--- Duffing parameters
Base.@kwdef struct Duf_Param
  u0::Vector = [0.0, 0.0] # initial condition
  p::Vector = [-1.0, 1.0, 1.0, 1,0, 1.0]# α, β, δ, γ, ω := stiffness, restore, damping, amptitude, angular frequency
  tspan::Tuple = (0.0, 10.0) # time span of simulation
  save_freq::Float32 = 0.1
end

#--- solve ODE

function solve_duffing(p)
  prob = ODEProblem(duffing,p.u0,p.tspan,p.p)
  sol = solve(prob,Tsit5(),saveat=p.save_freq)
  return sol
end

function animate_solution_two_lines(sol, pred)
  sol = Array(sol)
  pred = Array(pred)
  n,m = size(sol) # N by T
  @assert size(sol) == size(pred)
  p1 = plot(1, xlim=(-5,5), ylim=(-5,5), marker=2)
  p2 = plot(1, xlim=(-5,5), ylim=(-5,5), marker=2)
  l = @layout [ a ; b ]
  plot(p1, p2, layout=l)
  anim = @animate for j in 1:m
    v = sol[:,j]
    push!(p1, sol[1,j], sol[2,j])
    u = pred[:,j]
    push!(p2, pred[1,j], pred[2,j])
  end
  return anim
end
