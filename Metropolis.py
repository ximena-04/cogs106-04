import numpy as np
from scipy.stats import norm

class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.current = initialState
        self.accepted = 0
        self.proposed = 0
        self.stepSize = 0.1
        self.samples = []

    def accept(self, proposal):
        logAlpha = self.logTarget(proposal) - self.logTarget(self.current)
        alpha = np.exp(logAlpha)
        u = np.random.uniform()
        if u < alpha:
            self.accepted += 1
            self.current = proposal
            return True
        else:
            return False

    def adapt(self, blockLengths):
      targetAcceptanceRate = 0.4
      for blockLength in blockLengths:
        acceptanceCounter = 0
        proposalCounter = 0
        for i in range(blockLength):
            proposal = np.random.normal(self.current, self.stepSize)
            if self.accept(proposal):
                self.current = proposal
                acceptanceCounter += 1
            proposalCounter += 1
        acceptanceRate = acceptanceCounter / proposalCounter
        if acceptanceRate == 0:
            self.stepSize /= 2
        else:
            self.stepSize *= targetAcceptanceRate / acceptanceRate
      return self

    def sample(self, nsamples):
        for i in range(nsamples):
            proposal = np.random.normal(self.current, self.stepSize)
            if self.accept(proposal):
                self.current = proposal
            self.samples.append(self.current)
        self.samples = self.samples
        return self

    def summary(self):
        return {
            'mean': np.mean(self.samples),
            'c025': np.percentile(self.samples, 2.5),
            'c975': np.percentile(self.samples, 97.5),
        }
