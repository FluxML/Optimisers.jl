module OptimisersMLDataDevicesExt

using Optimisers: Optimisers
using MLDataDevices: get_device_type

# get_device_type, unlike get_device, works while Reactant is tracing.
function Optimisers._check_devices(arrays::AbstractVector{<:AbstractArray})
  try
    get_device_type(arrays)
  catch err
    err isa ArgumentError || rethrow()
    throw(ArgumentError("destructure needs all parameter arrays on one device. " * err.msg))
  end
  nothing
end

end
