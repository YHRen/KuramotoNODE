
using Flux, DiffEqFlux, OrdinaryDiffEq
using CuArrays
using IterTools: ncycle
using Formatting, Decimals
import Dates
using Flux: @epochs, @functor, @treelike
using Flux
using BSON: @save, @load
using  Pkg

include("Duffing.jl")

#--- solve Duffing
# α, β, δ, γ, ω := stiffness, restore, damping, amptitude, angular frequency
# chaotic:
# p = Duf_Param(u0=[1.0,0.0],p=[-1.0,1.0,0.1,0.9,1.0],tspan=(0.0,30.0))
p = Duf_Param(u0=[1.0,0.0],p=[-1.0,1.0,0.1,0.9,1.0],tspan=(0.0,30.0),save_freq=0.2)
sol=solve_duffing(p)
plt = plot(1, xlim=(-5,5), ylim=(-5,5), marker=2)
anim = @animate for i=1:length(sol)
  push!(plt, sol[1,i], sol[2,i])
end
gif(anim, "duffing.gif")

#--- NeuralODE
USE_GPU = true
RESUME = false
gt = Array(sol)[:,2:end]
u0 = p.u0

# dudt = Chain(
#   Dense(2,64,tanh),
#   Dense(64,128,tanh),
#   Dense(128,64,tanh),
#   Dense(64,16, tanh),
#   Dense(16,2)
# )
dudt = Chain(
  Dense(2,64,tanh),
  Dense(64,16, tanh),
  Dense(16,2)
)

if RESUME
  @load "best_duffing_model.bson" dudt
end

if USE_GPU
  gt  = gpu(gt)
  u0 = gpu(u0)
  dudt = gpu(dudt)
end

n_ode = NeuralODE(dudt, p.tspan, Tsit5(),saveat=p.save_freq, relative_err=1e-6, absolute_err=1e-6)
ps = Flux.params(n_ode)
opt = ADAM(0.001)

function pred_fn()
  pred = n_ode(u0)
  pred = Array(pred)
  return pred
end


function loss_fn()
  pred = pred_fn()
  loss = sum(abs2, pred.-gt)
  println(loss)
  return loss
end

t = range(p.tspan[1],p.tspan[2],length=size(gt)[2])
cb = function () #callback function to observe training
  cur_pred = pred_fn()
  pl = animate_solution_two_lines(sol, cur_pred)
  gif(pl)
  return pl
end

x = Iterators.repeated((),100)
#@time Flux.train!(loss_fn, ps, x, opt, cb=cb)

lowest_loss =typemax(Float32)
function my_custom_train!(loss, ps, data, opt, lowest_loss)
  tmp_loss = 0.0f0
  for d in data
    gs = gradient(ps) do
      training_loss = loss()
      #tmp_loss = training_loss
      # Insert what ever code you want here that needs Training loss, e.g. logging
      return training_loss
    end
    #if tmp_loss < lowest_loss
    #  lowest_loss = tmp_loss
    #  println("saving ", lowest_loss)
    #  @save "best_duffing_model.bson" dudt
    # end
    # insert what ever code you want here that needs gradient
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
    Flux.Optimise.update!(opt, ps, gs)
    #cb()
    # Here you might like to check validation set accuracy, and break out to do early stopping
  end
end

for e in 1:50
  my_custom_train!(loss_fn, ps, x, opt, lowest_loss)
  pl=cb()
  gif(pl,"duffing_snapshot_$(e).gif")
end

# savefig("snapshot_xxxiters_loss248_3441sec.png")
# savefig("snapshot_29loss.png")
