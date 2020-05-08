using Plots; gr()
using DifferentialEquations
using Distributions: Uniform
using LaTeXStrings
#--- define kuramoto model
function kuramotoTwoBody(du,u,p,t)
  θ₁, θ₂ = u
  K, ω₁, ω₂ = p
  k = K/length(u)
  du[1] = ω₁ + k*sin(θ₁-θ₂)
  du[2] = ω₂ + k*sin(θ₂-θ₁)
end

# general N-body kuramoto model
function kuramoto(du,u,p,t)
  K,Ω = p[1],p[2:end]
  k = K/length(u)
  for i = 1:length(u)
    du[i] = Ω[i] + k*sum(sin.(u[i].-u))
  end
  ## the following does not work
  #v = repeat(u, outer=[1,length(u)])
  #du = Ω + k*sum(sin.(v-transpose(v)), dims=2)
end

#--- define kuramoto related parameters
Base.@kwdef struct Param
  N::Int # system size
  K::Float32 # Coupling Strength
  L::Float32 # intrinsic frequency lower bound
  U::Float32 # intrinsic frequency upper bound
  tspan::Tuple = (0.0, 20.0) # time span of simulation
  save_freq::Float32 = 0.02
end

#--- define animation for kuramoto model
f2str(x,y=7) = string(x)[1:y]

function animate_solution(sol, p)
  sol = Array(sol)
  n,m = size(sol) # N by T
  anim = @animate for j in 1:m
    l = @layout [a{0.3h}; b ]
    plt = plot(layout=l)
    v = sol[:,j]

    plot!(plt[1], sin.(sol'), legend=:none)
    for i in 1:n
      scatter!(plt[1], [j], [sin.(v[i])], label="")
    end

    plot!(plt[2], xlims=(0,1), legend=:outertopleft)
    for i in 1:n
      plot!(plt[2], [v[i]], [1], st = :scatter,
        proj=:polar, marker=:o,
        label=latexstring("\$\\omega_{$(i)}={$(f2str(p[1+i]))}\$")
        )
    end
  end
  return anim
end

function animate_solution_lines(sol, p)
  sol = Array(sol)
  n,m = size(sol) # N by T
  anim = @animate for j in 1:m
    plot(sin.(sol'))
    for i in 1:n
      scatter!([j], [sin.(sol[i,j])], label="")
    end
  end
  return anim
end

function animate_solution_two_lines(sol, pred, p)
  sol = Array(sol)
  pred = Array(pred)
  n,m = size(sol) # N by T
  @assert size(sol) == size(pred)
  anim = @animate for j in 1:m
    l = @layout [a; b]
    plt = plot(layout=l)
    v = sol[:,j]
    plot!(plt[1], sin.(sol'), legend=:none)
    for i in 1:n
      scatter!(plt[1], [j], [sin.(v[i])], label="")
    end

    u = pred[:,j]
    plot!(plt[2], sin.(pred'), legend=:none)
    for i in 1:n
      scatter!(plt[2], [j], [sin.(u[i])], label="")
    end
  end
  return anim
end

function animate_solution_polar(sol, p)
  anim = @animate for i in 1:length(sol)
    plot([sol.u[i][1]], [1], seriestype = :scatter,
     proj=:polar, marker=:o, title="K="*string(p[1]),
     label=f2str(p[2]))
    for j in 2:size(sol)[1]
      plot!([sol.u[i][j]], [1], seriestype = :scatter,
        proj=:polar, marker=:o, label=f2str(p[1+j]))
    end
  end
  anim
end

#--- define kuramoto solver
function generate_random_parameters(k_param)
  # coupling strength, intrinsic freq. uniformly sampled from [L,U]
  p = [k_param.K; rand(Uniform(k_param.L,k_param.U),k_param.N)]
  return p
end

function solve_kuramoto(p, k_param)
  u0 = zeros(k_param.N)
  prob = ODEProblem(kuramoto,u0,k_param.tspan,p)
  sol = solve(prob,Tsit5(),saveat=k_param.save_freq)
  return sol
end
