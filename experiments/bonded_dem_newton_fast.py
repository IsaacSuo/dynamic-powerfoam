"""Optional drop-in Newton solver with the validated fused kernels.

The base Newton equation, iterations and stopping rules remain unchanged.
The running baseline imports NewtonBonds and is not switched to this class.
"""
from experiments.bonded_dem_newton import NewtonBonds
from experiments.bonded_dem_jvp_warp import WarpJacobian
from experiments.bonded_dem_precondition_warp import WarpDiagonal, WarpElastic
from experiments.bonded_dem_tangent_warp import WarpTangent


class FastNewtonBonds(NewtonBonds):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jacobian_vector = WarpJacobian(self)
        self._diagonal_blocks = WarpDiagonal(self)
        self.elastic_precondition = WarpElastic(self)
        if self.p.is_cuda:
            self.stiffness = WarpTangent(self)
