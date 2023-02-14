import random
import numpy as np
from scipy.interpolate import lagrange
from scipy.stats import norm


a = 0
b = 25

training = random.sample(range(a, b), 20)
testing = random.sample(range(a, b), 20)


y = [np.sin(x) for x in training]
y2 =np.array([np.sin(x) for x in testing])
y3 = np.array([np.sin(x + norm.rvs(size=1)[0]) for x in testing])

model = lagrange(training,y)

t_data = np.array([model(x) for x in testing])
print(sum(y2-t_data)/len(y2))
print(sum(y3-t_data)/len(y3))
# print(model)