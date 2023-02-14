import random
import numpy as np
from scipy.interpolate import lagrange
from scipy.stats import norm


a = 0
b = 25

training = random.sample(range(a, b), 20)
testing = random.sample(range(a, b), 20)

y = np.array([np.sin(x) for x in training])
y2 =np.array([np.sin(x) for x in testing])
y3 = np.array([np.sin(x) for x in testing])

ave_RMS = 0
ave_RMS2 = 0
for i in range(0,1000):
    training_zero_mean = training + norm.rvs(size=20)

    model = lagrange(training,y)

    model_zero_mean = lagrange(training_zero_mean,y)



    t_data = np.array([model(x) for x in testing])
    t_data2 = np.array([model_zero_mean(x) for x in testing])

    RMS = np.sqrt( (np.mean(y3-t_data)**2)/len(y2))

    RMS2 = np.sqrt( (np.mean(y3-t_data2)**2)/len(y3))

    ave_RMS+= RMS
    ave_RMS2 += RMS2

print(f"Average RMS of noiseless testing set for 1000 trials {ave_RMS/1000}")
print(f"Average RMS of zero mean noise testing set {ave_RMS2/1000}")

ave_RMS2 = 0
for i in range(0,1000):
    mean = random.randint(1,100)
    std = np.sqrt(mean)
    training_tainted = training + norm.rvs(mean, size=20)[0]
    
    y = np.array([np.sin(x) for x in training])


    model = lagrange(training_tainted,y)

    t_data = np.array([model(x) for x in testing])
    RMS2 = np.sqrt( (np.mean(y3-t_data)**2)/len(y3))
    ave_RMS2+=RMS2

print(f"RMS of noise injected testing set with means varrying from [1,100] the std will also vary because the std depends on the mean.\n {ave_RMS2/1000}")

