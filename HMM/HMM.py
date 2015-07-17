# -*- coding:utf-8 -*-
"""
@author: zhanghaidong
"""
import numpy as np

class HMM:
    def __init__(self, StatesNum, ObservationNum):
        self.StatesNum = StatesNum              # The number of hidden states
        self.ObservationNum = ObservationNum    # The number of observations
        self.TransitionProbs = np.zeros((self.StatesNum, self.StatesNum), np.float)
        self.EmssProbs = np.zeros((self.StatesNum, self.ObservationNum), np.float)
        self.InitProbs = np.zeros(self.StatesNum, np.float)

    def viterbi(self):
