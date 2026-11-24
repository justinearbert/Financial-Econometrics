# Lecture 6

import math, statistics, pandas as pd, matplotlib.pyplot as plt, numpy as np
import scipy.stats, statsmodels.api as sm
from arch import arch_model

# Ex.30
mu=0.00082; omega=0.000004; alpha=0.174862; beta=0.785715
sigmat2=omega+alpha*(-0.0197-mu)**2+beta*0.0099**2
print("sigmat = ",sigmat2**0.5)
q=1.281552
print("VaR = ",-mu+sigmat2**0.5*q)
q=1.644854
print("VaR = ",-mu+sigmat2**0.5*q)
q=2.3264348
print("VaR = ",-mu+sigmat2**0.5*q)

q=1.281552; alpha=0.1
print("ES = ",-mu+sigmat2**0.5/alpha/(2*math.pi)**0.5*math.exp(-q**2/2))
q=1.644854; alpha=0.05
print("ES = ",-mu+sigmat2**0.5/alpha/(2*math.pi)**0.5*math.exp(-q**2/2))
q=2.3264348; alpha=0.01
print("ES = ",-mu+sigmat2**0.5/alpha/(2*math.pi)**0.5*math.exp(-q**2/2))


# Ex.32
# a)
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1pAF02FLaGpwwklbLHKPT_dU7XDCdqCOx')
data.Date=pd.to_datetime(data.Date,format='%m/%d/%Y')
rt=data.SP500
model=arch_model(100*rt,vol='GARCH',p=1,q=1,mean='Constant',dist='t')
res=model.fit()
print(res.summary())
pars=res.params
pars['mu']=pars['mu']/100
pars['omega']=pars['omega']/10000
print(pars)

#b)
np.random.seed(1)
T=len(rt)
mu=pars.iloc[0]; omega=pars.iloc[1]; alpha=pars.iloc[2]; beta=pars.iloc[3]; nu=pars.iloc[4]
eps=np.random.standard_t(nu,T)/(nu/(nu-2))**0.5 # Step 1
rsim=[mu]*T; sigmat2=omega/(1-alpha-beta) # Step 2
for t in range(1,T):
    sigmat2=omega+alpha*(rsim[t-1]-mu)**2+beta*sigmat2
    rsim[t]=mu+sigmat2**0.5*eps[t]
#plt.plot(data.Date,rsim)
#plt.ylabel('Student t-GARCH(1,1) Simulated Returns')
#plt.ylim(-0.13,0.1) # to fix the range of y values between -0.13 and 0.1
#plt.show()

#plt.plot(data.Date,rt)
#plt.ylabel('S&P500 Returns')
#plt.ylim(-0.13,0.1) # to fix the range of y values between -0.13 and 0.1
#plt.show()

# Ex.33
# a)
model=arch_model(100*rt[1:T],x=100*rt[0:(T-1)],vol='GARCH',p=1,q=1,mean='ARX',dist='t')
res=model.fit()
print(res.summary())
pars=res.params
pars['Const']=pars['Const']/100
pars['omega']=pars['omega']/10000
print(pars)

print('Mean return on t if rt-1=0.01 = ',0.000975-0.04657*0.01)
print('Mean return on t if rt-1=0.10 = ',0.000975-0.04657*0.1)