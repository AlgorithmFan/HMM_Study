# -*- coding: utf-8 -*-
"""
@author: zhanghaidong
date: 2015/7/17
"""
import numpy as np

def normVector(vector):
    s = sum(vector)
    assert(s != 0)
    vector /= s
    return vector

def normMatrix(matrix):
    s = matrix.sum(axis=1)
    matrix = matrix/s[..., np.newaxis]
    return matrix

class HMM:
    def __init__(self, _A=None, _B=None, _I=None):
        self.TransProbs = _A
        self.InitProbs = _I
        self.EmssProbs =  _B
        if self.TransProbs is not None:
            self.StatesNum = self.TransProbs.shape[0]
        else:
            self.StatesNum = 0
        if self.EmssProbs is not None:
            self.ObservationNum = self.EmssProbs.shape[1]
        else:
            self.ObservationNum = 0

    def viterbi(self, observations):
        state_num = self.TransProbs.shape[0]
        T = observations.shape[0]
        delta = np.zeros((T, state_num), np.float)
        psi = np.zeros((T, state_num), np.int)
        for i in range(T):
            observation = observations[i]
            emss = (self.EmssProbs * observation[np.newaxis, ...]).sum(axis=1)
            if i == 0:
                delta[0,:] = emss * self.InitProbs
            else:
                score = (delta[i-1, :][..., np.newaxis] * self.TransProbs).transpose()
                delta[i, :] = np.max(score, axis=1) * emss
                psi[i, :] = np.argmax(score, axis=1)

        path = []
        path.insert(0, np.argmax(delta[-1, :]))
        for i in range(T-1, 0, -1):
            path.insert(0, psi[i, path[0]])
        return path



    def forwardbackward(self):
        pass

    def forward(self):
        pass


    def generate(self, StatesNum, ObservationsNum, Steps):
        self.StatesNum = StatesNum
        self.ObservationNum = ObservationsNum
        # Generate a initial probability
        self.InitProbs = np.random.rand(self.StatesNum)
        self.InitProbs = normVector(self.InitProbs)

        # Generate transformation probability
        self.TransProbs = np.random.rand(self.StatesNum, self.StatesNum)
        self.TransProbs = normMatrix(self.TransProbs)

        # Generate observation probability
        self.EmssProbs = np.random.rand(self.StatesNum, self.ObservationNum)
        self.EmssProbs = normMatrix(self.EmssProbs)

        def drawFrom(Probs):
            return np.where(np.random.multinomial(1, Probs) == 1)[0][0]

        observations = np.zeros(Steps)
        states = np.zeros(Steps)
        states[0] = drawFrom(self.InitProbs)
        observations[0] = drawFrom(self.EmssProbs[states[0], :])
        for t in range(1, Steps):
            states[t] = drawFrom(self.TransProbs[states[t-1], :])
            observations[t] = drawFrom(self.EmssProbs[states[t], :])
        return states, observations

if __name__ == '__main__':
    np.random.seed(1)
    mHMM = HMM()
    StatesNum = 3
    ObservationsNum = 4
    Steps = 10
    states, sequence = mHMM.generate(StatesNum, ObservationsNum, Steps)

    print 'The initial probability is '
    print mHMM.InitProbs
    print 'The transitive probaiblity is '
    print mHMM.TransProbs
    print 'The emission probability is '
    print mHMM.EmssProbs
    print 'The number of states is ', len(states)
    print states
    print 'The number of observations is ', len(sequence)
    print sequence
    observations = []
    for i in range(len(sequence)):
        temp = np.zeros(ObservationsNum)
        temp[sequence[i]] = 1
        observations.append(temp)
    # print 'The matrix of observations is '
    # print np.array(observations)
    print 'Viterbi algorithm.'
    observations = np.array(observations)
    path = mHMM.viterbi(observations)
    print path
    print states

    A = np.ones((3,3)) * 0.333333
    B = [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]
    B = np.array(B)
    I = np.ones(3) * 0.333333
    mHMM = HMM(A, B, I)
    sequence = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2])-1
    observations = []
    for i in range(len(sequence)):
        temp = np.zeros(2)
        temp[sequence[i]] = 1
        observations.append(temp)
    observations = np.array(observations)
    path = mHMM.viterbi(observations)
    print np.array(path) +1
