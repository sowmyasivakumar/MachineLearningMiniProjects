import numpy as np
from environment import MountainCar

a = np.array([])
print(np.append(a,0))

# choiceOptions = [0,1]
# choice = random.choices(choiceOptions, weights=(1-epsilon,epsilon ), k=1)
# random action

environment = MountainCar(mode="raw", fixed=True)
np_random = environment.np_random
value = np_random.normal()
