import sys
from environment import MountainCar
import numpy as np
import random


# def main(args):
#     pass

# <mode> <weight out>
# <returns out> <episodes> <max iterations> <epsilon> <gamma> <learning rate>
#
# if __name__ == "__main__":
#
#     mode = sys.argv[1]
#     weightsOut = sys.argv[2]
#     returnsOut = sys.argv[3]
#     episodes = int(sys.argv[4])
#
#     maxIterations = int(sys.argv[5])
#     epsilon = float(sys.argv[6])
#     gamma = float(sys.argv[7])
#     lr = float(sys.argv[8])
#
#
class LinearModel:
    def __init__(self, state_size, action_size,lr, indices):


            self.W =  np.zeros([state_size, action_size])
            self.lr = lr
            self.bias = 0
            self.indices = indices

    def predict(self, state) :
        currState = np.array(list(state.values()))

        currState = currState.reshape(len(state),1)
        Q = np.sum((currState * self.W), axis = 0)+self.bias
        return Q

    def update(self, state, action, target):

        currState = np.array(list(state.values()))

        # gradient(W)
        gradient = np.zeros_like(self.W)
        gradient[:, action] = currState



        #update rule
        self.W = self.W - target*gradient
        self.bias = self.bias - target*1





class QLearningAgent:
    def __init__(self, env, mode, gamma,lr, epsilon):
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

    def convertState(self, state):

        s = np.zeros((2048,))
        indices = list(state.keys())
        s[indices] = 1
        return dict(enumerate(s))

    def optimalPolicy(self, state, linearModel):


        Q = linearModel.predict(state)

        bestAction = np.argmax(Q)
        bestQ = Q[bestAction]

        return bestAction, bestQ

    def getState(self, state):

        if linearModel.indices == True:
            currstate =  self.convertState(state)
        else :
            currstate = state

        return currstate

    def get_action(self, state, linearModel) -> int:
        np_random = env.np_random
        if np_random.uniform(0,1) < epsilon:
            return np_random.choice(np.arange(self.env.action_space))
        #greedy action
        else :
            bestAction, bestQ = self.optimalPolicy(state, linearModel)
            return bestAction


    def train(self, episodes, max_iterations, linearModel):
        rewardList = []
        actionList = []

        for ith_episode in range(0, episodes):

            print('episode' + str(ith_episode))
            state = env.reset()
            state = self.getState(state)

            currState = np.array(list(state.values()))

            totalEpisodeRewards = 0
            done = False
            i = 0
            while not done and i < max_iterations:
                i = i+1
                #get action based on epsilon greedy method
                action = self.get_action(state, linearModel)
                actionList.append(action)

                #get next state, reward, and whether done = True
                nextState, reward, done = env.step(action)
                nextState = self.getState(nextState)
                # nextState = np.array(list(nextState.values()))

                #currentQ
                bestCurrentQ = self.optimalPolicy(state, linearModel)[1]


                #best future Q
                bestFutureQ = self.optimalPolicy(nextState, linearModel)[1]

                target = lr*(bestCurrentQ - (reward + gamma*bestFutureQ))
                linearModel.update(state, action, target)
                totalEpisodeRewards = totalEpisodeRewards + reward
                state = nextState


            rewardList.append(totalEpisodeRewards)


        return rewardList, actionList






# #initialize manually
mode = 'tile'
# fixed = 1
episodes = 400
maxIterations = 200
epsilon = 0.05
gamma = 0.99
lr = 0.00005
weightsOut = 'weights_out.OUT'
returnsOut = 'returns_out.OUT'




# initialise mountain car environment and get the state action space
env = MountainCar(mode)

#initialise params
if mode == 'raw':
    indices = False
elif mode == 'tile':
    indices = True

linearModel = LinearModel(env.state_space, env.action_space, lr,indices)


#initialise variables of the env
agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)

#train model
returns, actions = agent.train(episodes, maxIterations, linearModel)

# metricsOutString = linearModel.bias + '\n'
# for x in np.nditer(linearModel.W):
#     print(x)
#   # metricsOutString = metricsOutString + x + '\n'

weightsOutString = str(f'{linearModel.bias}\n')
for x in np.nditer(linearModel.W):
    weightsOutString = weightsOutString + str(f'{x}\n')

returnsOutString = ''
for ret in returns:
    returnsOutString = returnsOutString + str(f'{ret}\n')

with open(weightsOut, 'w') as f:
    f.write(weightsOutString)

with open(returnsOut, 'w') as f:
    f.write(returnsOutString)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


movingAverage = moving_average(np.array(returns), 25)
np.savetxt('movingavg.txt', movingAverage)
print(weightsOutString)
print(returnsOutString)
