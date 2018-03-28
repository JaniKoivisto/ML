import numpy as np
from actor import Actor

def qLearning(actor, policy, initState, discount_factor):
    Q = np.zeros((actor.nSp, actor.nA))
    iterations = 10000
    i = 0
    currentState = initState
     
    while i <= iterations:
        i += 1
        action = policy(currentState)
        new_state, reward = actor.takeAction(currentState, action)
        Q[currentState][action] = Q[currentState][action] + (reward + discount_factor * np.max(Q[new_state]) - Q[currentState][action])
        currentState = new_state
    return Q
    
def optimalPath(actor, Q):
    path = []
    initState = actor.S_start[0]
    path.append(initState)
    currentState = initState

    while not actor.isGoal(currentState):
        best_action = np.where (Q[currentState] == np.max(Q[currentState]))[0]
        next_state = actor.updateState(currentState, best_action[0])
        path.append(next_state)
        currentState = next_state
    return path

def make_random_policy(actor):
    def random_policy(state):
        return np.random.choice(actor.A, 1)[0]

    return random_policy

if __name__ == "__main__":
    discount_factor = 0.9
    actor = Actor()
    Q = qLearning(actor, make_random_policy(actor), actor.S_start[0], discount_factor)
    print(Q)
    path = optimalPath(actor, Q)
    print(list(map(actor.state2Coord, path)))
    