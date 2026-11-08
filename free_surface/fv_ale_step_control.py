"""Remember accepted dt; acceptance criteria remain in the production solver."""
import math


class RememberedStep:
    def __init__(self, saved=None):
        self.next_dt=None if saved is None else saved['next_dt']
        if self.next_dt is not None and (not math.isfinite(self.next_dt) or self.next_dt<=0):
            raise ValueError('invalid remembered time step')

    def propose(self, ceiling):
        return ceiling if self.next_dt is None else min(ceiling,self.next_dt)

    def accept(self, dt, retries, boundary_limited=False):
        # A shortened output interval is not evidence of an accuracy limit.
        if boundary_limited and retries==0:return
        self.next_dt=dt*(1.25 if retries==0 else 1.)

    def packed(self):
        return {'next_dt':self.next_dt}
