import numpy as np

class Metropolis:
    def __init__(self, logTarget, initialState, stepSize=0.1):
        self.logTarget = logTarget
        self.state = initialState
        self.stepSize = stepSize
        self.samples = []
        self.acceptanceRate = 0.4
