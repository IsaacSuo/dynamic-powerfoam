"""Remember a successful subdivision instead of retrying a failed large dt."""


class AdaptiveIntervals:
    def __init__(self, solver):
        self.solver = solver
        self.depth = 0
        self.easy_intervals = 0

    def advance(self, dt, load):
        work, records = 0., []
        for _ in range(2**self.depth):
            value, info = self.solver.advance_interval(dt/(2**self.depth), load, depth=self.depth)
            work += value; records.extend(info)
        self.depth = max(item['subdivision_depth'] for item in records)
        if max(item['nonlinear_iterations'] for item in records) <= 3:
            self.easy_intervals += 1
        else:
            self.easy_intervals = 0
        if self.easy_intervals >= 8 and self.depth:
            self.depth -= 1
            self.easy_intervals = 0
        return work, records
