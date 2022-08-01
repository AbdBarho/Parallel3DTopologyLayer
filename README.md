# Parallel3DTopologyLayer
A 3D slice-wise CPU-parallelized implementation on top of [TopologyLayer](https://github.com/bruel-gabrielsson/TopologyLayer).

## Classes

### Topo3DLoss
computes toplogy loss on each slice on the first axis individually.

### Topo3DLossRotatingAxis
computes toplogy loss on a axis `epoch % 3`

### Topo3DLossAllAxes
computes toplogy loss on all axes, as an average of the individual axis





This was used as part of our submission for the PARSE 2022 challenge.
