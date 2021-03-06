using Flux, DiffEqFlux, OrdinaryDiffEq
using CuArrays
using IterTools: ncycle
using Formatting, Decimals
import Dates
using Flux: @epochs, @functor, @treelike
using Flux
using BSON: @save, @load
using  Pkg

include("LotkaVolterra.jl")

#--- solve spiral
p = Spiral_Param()
sol=solve_spiral(p)
plt = plot(1, xlim=(-2,2), ylim=(-2,2), marker=2)
anim = @animate for i=1:101
  push!(plt, sol[1,i], sol[2,i])
end
gif(anim, "spiral.gif")


#--- solve LV
p = LV_Param()
sol = solve_lotka_volterra(p)

plot(sol)
print(size(sol))

#--- NeuralODE
USE_GPU = false
gt = Array(sol)[:,2:end]
u0 = p.u0

dudt = Chain(
  Dense(2,64,tanh),
  Dense(64,128,tanh),
  Dense(128,64,tanh),
  Dense(64,16, tanh),
  Dense(16,2)
)

if RETRAIN
  @load "best_model.bson" dudt
end
if USE_GPU
  gt  = gpu(gt)
  u0 = gpu(u0)
  dudt = gpu(dudt)
end

n_ode = NeuralODE(dudt, p.tspan, Tsit5(),saveat=p.save_freq, relative_err=1e-7, absolute_err=1e-8)
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
  #println(loss)
  return loss
end

t = range(p.tspan[1],p.tspan[2],length=size(gt)[2])

cb = function () #callback function to observe training
  pl = plot(sol)
  cur_pred = pred_fn()
  scatter!(pl,t,(cur_pred[1,:]),label="prediction1")
  scatter!(pl,t,(cur_pred[2,:]),label="prediction2")
  display(plot(pl))
end

x = Iterators.repeated((),10)
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
    #  @save "best_model.bson" dudt
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
  cb()
  savefig("snapshot_$(e).png")
end


savefig("snapshot_xxxiters_loss248_3441sec.png")
savefig("snapshot_29loss.png")
