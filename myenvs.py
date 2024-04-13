"""A simulation environment"""
import os
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from myutils import get_OTC_path 


class TradingEnv(gym.Env):

    def __init__(self, filename, train_flag = False, dg_random_seed=1, num_sim=4001, init_ttm=100):

        if train_flag:
            if os.path.exists('history/training_history'+filename+'.csv'):
                print("Training file already exists, exiting env class")
                self.exited = True
                return 
            self.exited = False
        else:  
            if os.path.exists('history/testing_history'+filename+'.csv'):
                print("Testing file already exists, exiting env class")
                self.exited = True
                return
            self.exited = False
        
        self.step = self.step_profit_loss
        # other attributes
        self.num_path = num_sim 
        self.num_period = init_ttm 
        
        # get OTC prices paths and OP
        self.pc_all, self.p2_all, self.p1_all, self.ch_all, self.cp_all, self.w1_all, self.w2_all, self.Vo = get_OTC_path(T_training = self.num_period, np_seed=dg_random_seed, T_testing = 1, nmax = num_sim) 

        # track the index of simulated path in use
        self.sim_episode = -1

        # track time step within an episode  
        self.t = None
        self.state = []        
        self.sim_episode = (self.sim_episode + 1) % self.num_path

        self.t = 0
        s1 = self.w1_all[self.sim_episode, self.t]
        s2 = self.w2_all[self.sim_episode, self.t]
        
        action = np.zeros((4))
        self.action_space = spaces.Box(low = np.array([0, 0, 0, 0]),
                                   high = np.array([min(98107,s1), min(max(98107 - action[0], 0), s2), max(s1 - action[0], 0), max(s2 - action[1],0)]),
                                   dtype=np.float32)   
        
        self.num_state = 7

        # seed and start
        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # repeatedly go through available simulated paths (if needed)
        self.sim_episode = (self.sim_episode + 1) % self.num_path
        self.t = 0
        pc = self.pc_all[self.sim_episode, self.t]
        p2 = self.p2_all[self.sim_episode, self.t]
        p1 = self.p1_all[self.sim_episode, self.t]
        ch = self.ch_all[self.sim_episode, self.t]
        cp = self.cp_all[self.sim_episode, self.t]
        s1 = self.w1_all[self.sim_episode, self.t]
        s2 = self.w2_all[self.sim_episode, self.t]
        V_opt = self.Vo[0,self.sim_episode]  
   
        if np.isnan(s2).any():
            print('reset s2 has nan') 

        self.state = [pc, p2, p1, ch, cp, s1, s2] 
        
        if np.isnan(self.state).any():
            print('reset self.state has nan')          
            self.state[np.isnan(self.state)] = 0
            print('self.state has nan reset', self.state)       
        
        return self.state, V_opt 

    

    def step_profit_loss(self, x, action):
        """
        profit loss period reward
        """
 
        pc = x[0,0]
        p2 = x[0,1]
        p1 = x[0,2]
        ch = x[0,3]
        cp = x[0,4]
        s1 = x[0,5] 
        s2 = x[0,6] # this period's w2+holding of UWH from last period
         

        def reward_function_const(state, action):
            
            cash_flow0 = 0
            alpha = 0.154
            q = 98107
            
            y1 = action[0]
            y2 = action[1]
            z1 = action[2]
            z2 = action[3]
            pc = state[0,0]
            p2 = state[0,1]
            p1 = state[0,2]
            ch = state[0,3]
            cp = state[0,4]
            s1 = state[0,5] 
            s2 = state[0,6] # this period's w2+holding of UWH from last period
            w1 = self.w1_all[self.sim_episode, self.t]
            w2 = self.w2_all[self.sim_episode, self.t]
            
            if y1 + z1 > s1 + 1 or y2 + z2 > s2 + 1 or y1 + y2 > q + 1:                 
                print('violation y1 + y2 - q -1 = ', y1 + y2 - q -1)
                print('violation y1 + z1 - s1 -1 = ', y1 + z1 - s1 -1)
                print('violation y2 + z2 - s2 -1 = ',  y2 + z2 - s2 -1)
                
                return 0
            
            else: 
                cash_flow0 = pc*(y2+(1-alpha)*y1) + p1*(s1 - y1 - z1) + p2*(s2 - y2 - z2) - ch*z1 - ch*z2 - 1*cp*(q - y1 -y2) -ch*(w1+w2)
                
                return cash_flow0

        reward = reward_function_const(x, action)
        # update time
        self.t = self.t + 1

        
        pc = self.pc_all[self.sim_episode, self.t]
        p2 = self.p2_all[self.sim_episode, self.t]
        p1 = self.p1_all[self.sim_episode, self.t]
        ch = self.ch_all[self.sim_episode, self.t]
        cp = self.cp_all[self.sim_episode, self.t]
        w1 = self.w1_all[self.sim_episode, self.t]
        w2 = self.w2_all[self.sim_episode, self.t]
    
        s1 = w1
        s2 = w2 + action[2] + action[3] 

        self.state = [pc, p2, p1, ch, cp, s1, s2]  

        if np.isnan(self.state).any():
            print('reset self.state has nan')          
            self.state[np.isnan(self.state)] = 0
            print('self.state has nan reset', self.state) 
        # if tomorrow is end of episode
        if self.t == self.num_period - 1:
            done = True
            reward = reward  
        else:
            done = False
            reward = reward  

        # for other info later
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info
