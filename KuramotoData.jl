include("Kuramoto.jl")


#--- demostrate kuramoto system using ode solver
function create_a_demo(k_param)
  p = generate_random_parameters(k_param)
  sol = solve_kuramoto(p, k_param)
  anim = animate_solution(sol, p)
  gif(anim)
  return anim
end

#--- create a training dataset
function create_training_data(data_size, k_param)
  time_series_len = convert(Int32,k_param.tspan[2]/k_param.save_freq+1)
  print(time_series_len)
  x_data_buffer = zeros(Float32, k_param.N, data_size)
  y_data_buffer = zeros(Float32, k_param.N, time_series_len, data_size) # k_param.time_span
  for i in 1:data_size
    p = generate_random_parameters(k_param)
    y = Array(solve_kuramoto(p, k_param))
    convert(Array{Float32}, p)
    convert(Array{Float32}, y)
    x_data_buffer[:,i] = p[2:end]
    y_data_buffer[:,:,i] = y
  end
  return x_data_buffer, y_data_buffer
end
