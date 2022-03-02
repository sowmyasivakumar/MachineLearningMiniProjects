from environment import MountainCar
import numpy as np

environment = MountainCar(mode="tile")
state = environment.reset()
s = np.zeros((2048,))
indices = list(state.keys())
s[indices] = 1
print(dict(enumerate(s)))



