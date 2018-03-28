import numpy as np

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

class Actor:
    def __init__(self):
        self.world, self.rewards = Actor.createExampleWorld ()

        self.shape = self.world.shape
        self.MAX_Y = self.shape[0]
        self.MAX_X = self.shape[1]

        # State space
        self.nSp = self.MAX_X * self.MAX_Y
        self.S = []
        self.Sp = []
        self.S_start = []

        # Action set
        self.A = [NORTH, SOUTH, WEST, EAST]
        self.nA = len (self.A)

        self.T = {} # Transition
        self.R = {} # Reward

        for state in range (self.nSp):
            self.T[state] = {a : [] for a in self.A}
            self.R[state] = {a : [] for a in self.A}

            if self.isGoal (state):
                self.Sp.append (state)

                for action in self.A:
                    new_state_a = self.updateState (state, action)

                    x, y = self.state2Coord (new_state_a)
                    reward = self.rewards[self.world[y, x]]
                                                                                                                
                    self.T[state][action] = new_state_a
                    self.R[state][action] = reward
            else:
                self.S.append (state)

                for action in self.A:
                    new_state_a = self.updateState (state, action)

                    x, y = self.state2Coord (new_state_a)
                    reward = self.rewards[self.world[y, x]]                                                  
                                                        
                    self.T[state][action] = new_state_a
                    self.R[state][action] = reward

            if self.isStart (state):
                self.S_start.append (state)

        self.Sp = self.Sp + self.S
    
    def takeAction (self, state, action):
        new_state_a = self.T[state][action]
        reward = self.R[state][action]

        return new_state_a, reward

    def updateState (self, state, action):
        xy = np.array (self.state2Coord (state), dtype = "int")

        if action == NORTH:
            new_xy = xy + [0, -1]
        elif action == SOUTH:
            new_xy = xy + [0, 1]
        elif action == WEST:
            new_xy = xy + [-1, 0]
        elif action == EAST:
            new_xy = xy + [1, 0]
        else:
            raise ValueError ("Invalid action %d." % action)

        if self.isOutside (new_xy):
            return state
        elif self.isObstacle (new_xy):
            return state
        else:
            return self.coord2State (new_xy)
    
    def isGoal (self, state):
        x, y = self.state2Coord (state)
        return self.world[y, x] == "G"

    def isStart (self, state):
        x, y = self.state2Coord (state)
        return self.world[y, x] == "S"

    def isOutside (self, xy):
        y, x = xy[1], xy[0]
        return y < 0 or x < 0 or y >= self.MAX_Y or x >= self.MAX_X
    
    def isObstacle (self, xy):
        y, x = xy[1], xy[0]
        return self.world[y, x] == 'o'
    
    def state2Coord (self, state):
        yx = np.unravel_index (state, self.shape, order ="C")
        return (yx[1], yx[0])

    def coord2State (self, xy):
        return np.ravel_multi_index ((xy[1], xy[0]), self.shape, order = "C")

    def createExampleWorld ():
        rewards = {"G": 0.0, ".": -1.0, "o": -1.0, "S": -1.0}
        world = np.array (
            [[".", ".", ".", ".", ".", "."],
             [".", "o", ".", ".", "S", "."],
             [".", "o", "o", ".", ".", "."],
             [".", ".", "o", ".", ".", "."],
             [".", ".", "o", "o", "o", "o"],
             [".", ".", ".", ".", ".", "G"]])
        return world, rewards
