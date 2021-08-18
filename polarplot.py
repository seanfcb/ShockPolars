import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optim

x = 3

def oblique(Mx,sigma,gamma):
    Mxn   = Mx*np.sin(sigma)

    delta_top    = (Mx**2*np.sin(sigma)*np.cos(sigma) - np.cos(sigma)/np.sin(sigma))
    delta_bottom = 0.5*(gamma+1)*Mx**2 - (Mx*np.sin(sigma))**2 + 1
    delta = np.arctan(delta_top/delta_bottom)
    Pr    = (2*gamma*Mx**2*(np.sin(sigma))**2-(gamma-1))/(gamma+1)
    Myn   = np.sqrt((2+(gamma-1)*Mxn**2)/(2*gamma*Mxn**2-(gamma-1)))
    My    = Myn/(np.sin(sigma-delta))
    return(delta, Pr, My)

def obliqued(Mx, delta, gamma):
  sigma = 0
  Pr = 0
  My = 0
  return(sigma, Pr, My)
  
def double_ramp(gamma=None, M1=None, delta_ramp=None):
  gamma = gamma or 1.4
  M1    = M1    or 1.5
  delta_ramp = delta_ramp or 7.5*np.pi/180

  
  #shock polar for first shock
  sigma1_min = np.arcsin(1/M1)
  sigma1_max = 90*np.pi/180
  sigma1 = np.linspace(sigma1_min,sigma1_max,20)
  delta1, Pr1, M2 = oblique(M1, sigma1, gamma)

  delta_ramp = delta1[2] #random until you get obliqued() working
  Pr_ramp    = Pr1[2]
  M_ramp     = M2[2]

  sigma2_min = np.arcsin(1/M_ramp)
  sigma2_max = 90*np.pi/180
  sigma2 = np.linspace(sigma2_min,sigma2_max,20)
  delta2, Pr2, M3 = oblique(M_ramp, sigma2, gamma)
  Pr2 = Pr2*Pr_ramp
  delta2 = delta_ramp - delta2
  
  plt.plot(delta1*180/np.pi, Pr1, '-r')
  plt.plot(delta2*180/np.pi, Pr2, '-b')
  plt.plot(np.array([1,1])*delta_ramp*180/np.pi, np.array([Pr1[0],Pr1[-1]]),'--')
  plt.show()
  
def basicplot(gamma=None, M1=None):
  
  gamma = gamma or 1.4
  M1    = M1    or 1.5
  sigma1_min = np.arcsin(1/M1)
  sigma1_max = 90*np.pi/180

  sigma1 = np.linspace(sigma1_min,sigma1_max,20)
  delta1, Pr1, M2 = oblique(M1, sigma1, gamma)
  delta_ramp = delta1[5]
  
  #M_ramp     = M2[5]
  #Pr_ramp    = Pr1[5]
  
  M_ramp = 2*M1
  sigma2_min = np.arcsin(1/M_ramp)
  sigma2_max = 90*np.pi/180
  sigma2 = np.linspace(sigma2_min,sigma2_max,20)
  delta2, Pr2, M3 = oblique(M_ramp, sigma2, gamma)

  #
  #plt.plot(delta1*180/np.pi,Pr1)
  #plt.plot(delta1*180/np.pi, M2)
  #plt.plot((delta_ramp-delta2)*180/np.pi,Pr2*Pr_ramp)
  plt.plot(sigma1*180/np.pi, delta1*180/np.pi,'-')
  plt.plot(sigma2*180/np.pi, delta2*180/np.pi,'--')

  plt.show()
  
  plt.plot(delta1*180/np.pi,Pr1,'-')
  plt.plot(delta2*180/np.pi,Pr2,'--')
  plt.show()
  
if __name__ == "__main__":
  basicplot()
