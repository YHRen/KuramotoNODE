using Flux, DiffEqFlux, OrdinaryDiffEq
using Flux
using CuArrays
using IterTools: ncycle
using Formatting, Decimals
import Dates
using  Pkg

include("Kuramoto.jl")
include("KuramotoData.jl")
#--- define kuramoto related parameters
k_param = Param(N=3,K=1.0,L=0.2,U=1.0,tspan=(0.0f0, 20.0f0), save_freq=0.1f0)
data_size = 200
train_x, train_y = create_training_data(data_size, k_param)

###---- assert Flux data loader
bsz = 100
time_series_len = convert(Int32, k_param.tspan[2]/k_param.save_freq+1)
train_data_loader = Flux.Data.DataLoader(train_x, train_y, batchsize=bsz, shuffle=true)
for (x,y) in train_data_loader
  @assert size(x) == (k_param.N, bsz)
  @assert size(y) == (k_param.N, time_series_len, bsz)
end
println(size(train_x), size(train_y))

#--- NeuralODE

CuArrays.allowscalar(false)
dudt = Chain(
  Dense(k_param.N,50,tanh),
  Dense(50,100,tanh),
  Dense(100,k_param.N)
)

n_ode = NeuralODE(dudt, k_param.tspan,
          Tsit5(),saveat=k_param.save_freq)
ps = Flux.params(n_ode)
opt = ADAM(0.001)

function pred_fn(x)
  #x = gpu(x)
  pred = n_ode(x)
  pred = Array(pred)
  pred = permutedims(pred, [1,3,2]) # to match y
  return pred
end

module Cnt
  global idx = 1
end

function test_random()
  test_p = generate_random_parameters(k_param)
  gt_sol = solve_kuramoto(test_p, k_param)
  pred_sol = n_ode(test_p[2:end])
  anim = animate_solution_two_lines(gt_sol, pred_sol, test_p)
  timestamp = Dates.value(Dates.now())
  #gif(anim, "./$(timestamp).gif")
  gif(anim, "./$(Cnt.idx).gif")
  @eval Cnt idx += 1
  return anim
end

function loss_fn(x, y)
  loss = sum(abs2, pred_fn(x).-y)
  println(loss)
  return loss
end
@time Flux.train!(loss_fn, ps, ncycle(train_data_loader, 5000), opt,
 cb=Flux.throttle(test_random, 60))

#--- test train loader
for (x,y) in train_data_loader
  print(size(x))
  @time pred = n_ode(x)
  pred = Array(pred)
  pred = permutedims(pred, [1,3,2]) # to match y
  @assert size(pred) == size(y)
  print(size(pred), size(y))
end

#--- TODO: try to learn on one time series only.
