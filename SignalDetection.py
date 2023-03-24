import numpy
from scipy.special import ndtri
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SignalDetection:
  def __init__(self, hits, misses, falseAlarms, correctRejections):
    self.hits = hits
    self.misses = misses
    self.falseAlarms = falseAlarms
    self.correctRejections = correctRejections
  
  def hitRate(self):
    self.__hr = (self.hits / (self.hits + self.misses))
    return self.__hr

  def falseAlarmRate(self):
    self.__far = (self.falseAlarms / (self.falseAlarms + self.correctRejections))
    return self.__far

  def d_prime(self):
    self.__dprime = (ndtri(self.hitRate()) - ndtri(self.falseAlarmRate()))
    return self.__dprime
  
  def criterion(self):
    self.__criterion = (-0.5 * (ndtri(self.hitRate()) + ndtri(self.falseAlarmRate())))
    return self.__criterion

  def threshold(self):
    return self.criterion() + (self.d_prime() / 2)

  def __add__(self, other):
    return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + other.falseAlarms, self.correctRejections + other.correctRejections)

  def __mul__(self, scalar):
    return SignalDetection(self.hits * scalar, self.misses * scalar, self.falseAlarms * scalar, self.correctRejections * scalar)
  
  @staticmethod
  def plot_roc(sdtList):
    x_vals = []
    y_vals = []
    for classObj in sdtList:
      classObj.hitRate()
      classObj.falseAlarmRate()
      newY = classObj.__hr
      newX = classObj.__far
      x_vals.append(newX)
      y_vals.append(newY)

    compX = [0, 1]
    compY = [0, 1]

    plt.plot(compX, compY, linestyle='dashed', color='k', label="x = y")
    plt.scatter(x_vals, y_vals, color='k')

    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal', adjustable='box')
    return 

  def plot_sdt(self):
    if 0 < self.d_prime():
      lowerBound = (- 5)
      upperBound = (self.d_prime() + 5)
    if self.d_prime() < 0: 
      lowerBound = (self.d_prime() - 5)
      upperBound = 5
    
    range = numpy.arange(lowerBound, upperBound, 0.01)
    signal = norm.pdf(range, (self.d_prime()), 1)
    noise = norm.pdf(range, 0, 1)

    peakX = [0, self.d_prime()]
    peakY = [(max(signal)), max(noise)] 

    plt.plot(range, signal, label="S", color='g')
    plt.plot(range, noise, label="N", color='r')
    plt.axvline(x=(self.threshold()), label="C", color='c')
    plt.plot(peakX,peakY,label='D', color='b')
  
    plt.xlabel("Signal Strength")
    plt.ylabel("Probability")
    plt.title("Signal Detection Theory (SDT) Plot")
    plt.legend(loc="upper right")
    plt.show()
    return

  @staticmethod
  def simulate(dprime, criteriaList, signalCount, noiseCount):
    sdtList = []
    for i in range(len(criteriaList)):
      k = criteriaList[i] + (dprime / 2)
      hr = 1 - (norm.cdf((k - dprime)))
      far = 1 - (norm.cdf(k))
      nSig = signalCount
      nNoi = noiseCount
      
      hits = numpy.random.binomial(nSig, hr)
      misses = nSig - hits
      fa = numpy.random.binomial(nNoi, far)
      cr = nNoi - fa 

      sdtObj = SignalDetection(hits, misses, fa, cr)
      sdtList.append(sdtObj)
    return sdtList

  def nLogLikelihood(self, hitRate, falseAlarmRate):
    nHLog = (self.hits * -1) * (numpy.log(hitRate))
    mLog = self.misses * (numpy.log((1 - hitRate)))
    fLog = self.falseAlarms * (numpy.log(falseAlarmRate))
    rLog = self.correctRejections * (numpy.log((1 - falseAlarmRate)))

    ell =  nHLog - mLog - fLog - rLog
    return ell 

  @staticmethod
  def rocCurve(falseAlarmRate, a):
    inner = (a + (norm.ppf(falseAlarmRate)))
    hitRate = norm.cdf(inner)
    return hitRate

  @staticmethod
  def rocLoss(a, sdtList):
    L = 0
    for obj in sdtList:
      far = obj.falseAlarmRate()
      hrPred = norm.cdf(a + (norm.ppf(far)))
      nLLL = obj.nLogLikelihood(hrPred, far)
      L += nLLL
    return L

  @staticmethod
  def fit_roc(sdtList):
    point = numpy.random.normal()
    roc_func = minimize(fun = SignalDetection.rocLoss, x0 = point, args = sdtList)
    
    scatterplot = SignalDetection.plot_roc(sdtList)

    x_vals = numpy.arange(0,1,0.01)
    y_vals = SignalDetection.rocCurve(x_vals, roc_func.x[0])

    plt.plot(x_vals, y_vals,label='fitted curve', color='r')
    plt.legend(loc="lower right")
    plt.ylabel("Hit Rate")
    plt.xlabel("False Alarm Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.show()
    return roc_func.x[0]
