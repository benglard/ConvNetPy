from util import *
from net import Net
from trainers import Trainer
from vol import Vol

"""
An agent is in state0 and does action0
environment then assigns reward0 and provides new state, state1
Experience nodes store all this information, which is used in the
Q-learning update step
"""

class Experience(object):
    
    def __init__(self, state0, action0, reward0, state1):
        self.state0 = state0
        self.action0 = action0
        self.reward0 = reward0
        self.state1 = state1

"""
A Brain object does all the magic.
over time it receives some inputs and some rewards
and its job is to set the outputs to maximize the expected reward
"""

class Brain(object):

    def __init__(self, num_states, num_actions, opt={}):
        """
        in number of time steps, of temporal memory
        the ACTUAL input to the net will be (x,a) temporal_window times, and followed by current x
        so to have no information from previous time step going into value function, set to 0.
        """
        self.temporal_window = getopt(opt, 'temporal_window', 1)

        """size of experience replay memory"""
        self.experience_size = getopt(opt, 'experience_size', 30000)

        """number of examples in experience replay memory before we begin learning"""
        self.start_learn_threshold = getopt(opt, 'start_learn_threshold',
                                            int(min(self.experience_size * 0.1, 1000)))

        """gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]"""
        self.gamma = getopt(opt, 'gamma', 0.8)

        """number of steps we will learn for"""
        self.learning_steps_total = getopt(opt, 'learning_steps_total', 100000)

        """how many steps of the above to perform only random actions (in the beginning)?"""
        self.learning_steps_burnin = getopt(opt, 'learning_steps_burnin', 3000)

        """what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end"""
        self.epsilon_min = getopt(opt, 'epsilon_min', 0.05)

        """what epsilon to use at test time? (i.e. when learning is disabled)"""
        self.epsilon_test_time = getopt(opt, 'epsilon_test_time', 0.01)

        """
        advanced feature. Sometimes a random action should be biased towards some values
        for example in flappy bird, we may want to choose to not flap more often
        """
        if 'random_action_distribution' in opt:
            #this better sum to 1 by the way, and be of length this.num_actions
            self.random_action_distribution = opt['random_action_distribution']

            if len(self.random_action_distribution) != num_actions:
                print 'TROUBLE. random_action_distribution should be same length as num_actions.'

            a = self.random_action_distribution
            s = sum(a)
            if abs(s - 1.0) > 0.0001:
                print 'TROUBLE. random_action_distribution should sum to 1!'
            else:
                self.random_action_distribution = []

        """
        states that go into neural net to predict optimal action look as
        x0,a0,x1,a1,x2,a2,...xt
        this variable controls the size of that temporal window. Actions are
        encoded as 1-of-k hot vectors
        """
        self.net_inputs = num_states * self.temporal_window + num_actions * self.temporal_window + num_states        
        self.num_states = num_states
        self.num_actions = num_actions
        self.window_size = max(self.temporal_window, 2) #must be at least 2, but if we want more context even more
        self.state_window = zeros(self.window_size)
        self.action_window = zeros(self.window_size)
        self.reward_window = zeros(self.window_size)
        self.net_window = zeros(self.window_size)

        #create [state -> value of all possible actions] modeling net for the value function
        layers = []
        if 'layers' in opt:
            """
            this is an advanced usage feature, because size of the input to the network, and number of
            actions must check out. 
            """
            layers = opt['layers']

            if len(layers) < 2:
                print 'TROUBLE! must have at least 2 layers'
            if layers[0]['type'] != 'input':
                print 'TROUBLE! first layer must be input layer!'
            if layers[-1]['type'] != 'regression':
                print 'TROUBLE! last layer must be input regression!'
            if layers[0]['out_depth'] * layers[0]['out_sx'] * layers[0]['out_sy'] != self.net_inputs:
                print 'TROUBLE! Number of inputs must be num_states * temporal_window + num_actions * temporal_window + num_states!'
            if layers[-1]['num_neurons'] != self.num_actions:
                print 'TROUBLE! Number of regression neurons should be num_actions!'
        else:
            #create a very simple neural net by default
            layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': self.net_inputs})
            if 'hidden_layer_sizes' in opt:
                #allow user to specify this via the option, for convenience
                for size in opt['hidden_layer_sizes']:
                    layers.append({'type': 'fc', 'num_neurons': size, 'activation': 'relu'})
            layers.append({'type': 'regression', 'num_neurons': self.num_actions}) #value function output

        self.value_net = Net(layers)

        #and finally we need a Temporal Difference Learning trainer!
        trainer_ops_default = {'learning_rate': 0.01, 'momentum': 0.0, 'batch_size': 64, 'l2_decay': 0.01}
        tdtrainer_options = getopt(opt, 'tdtrainer_options', trainer_ops_default)
        self.tdtrainer = Trainer(self.value_net, tdtrainer_options)

        #experience replay
        self.experience = []

        #various housekeeping variables
        self.age = 0            #incremented every backward()
        self.forward_passes = 0 #incremented every forward()
        self.epsilon = 1.0      #controls exploration exploitation tradeoff. Should be annealed over time
        self.latest_reward = 0
        self.last_input_array = []
        self.average_reward_window = Window(1000, 10)
        self.average_loss_window = Window(1000, 10)
        self.learning = True

    def random_action(self):
        """
        a bit of a helper function. It returns a random action
        we are abstracting this away because in future we may want to 
        do more sophisticated things. For example some actions could be more
        or less likely at "rest"/default state.
        """

        if len(random_action_distribution) == 0:
            return randi(0, self.num_actions)
        else:
            #okay, lets do some fancier sampling
            p = randf(0, 1.0)
            cumprob = 0.0
            for k in xrange(self.num_actions):
                cumprob += self.random_action_distribution[k]
                if p < cumprob:
                    return k

    def policy(self, s):
        """
        compute the value of doing any action in this state
        and return the argmax action and its value
        """

        V = Vol(s)
        action_values = self.value_net.forward(V)
        weights = action_values.w
        max_val = max(weights)
        max_k = weights.index(maxval)
        return {
            'action': max_k,
            'value': max_val
        }

    def getNetInput(self, xt):
        """
        return s = (x,a,x,a,x,a,xt) state vector
        It's a concatenation of last window_size (x,a) pairs and current state x
        """

        w = []
        w.extend(xt) #start with current state
        #and now go backwards and append states and actions from history temporal_window times
        n = self.window_size
        for k in xrange(self.temporal_window):
            index = n - 1 - k
            w.extend(self.state_window[index]) #state

            #action, encoded as 1-of-k indicator vector. We scale it up a bit because
            #we dont want weight regularization to undervalue this information, as it only exists once
            action1ofk = zeros(self.num_actions)
            action1ofk[index] = 1.0 * self.num_states
            w.extend(action1ofk)

        return w

    def forward(self, input_array):
        self.forward_passes += 1
        self.last_input_array = input_array

        # create network input
        action = None
        if self.forward_passes > self.temporal_window:
            #we have enough to actually do something reasonable
            net_input = self.getNetInput(input_array)
            if self.learning:
                #compute epsilon for the epsilon-greedy policy
                self.epsilon = min(
                    1.0,
                    max(
                        self.epsilon_min,
                        1.0 - \
                        (self.age - self.learning_steps_burnin) / \
                        (self.learning_steps_total - self.learning_steps_burnin)
                    )
                )
            else:
                self.epsilon = self.epsilon_test_time #use test-time value
            
            rf = randf(0, 1)
            if rf < self.epsilon:
                #choose a random action with epsilon probability
                action = self.random_action()
            else:
                #otherwise use our policy to make decision
                maxact = self.policy(net_input)
                action = maxact['action']
        else:
            #pathological case that happens first few iterations
            #before we accumulate window_size inputs
            net_input = []
            action = self.random_action()

        #remember the state and action we took for backward pass
        self.net_window.pop(0)
        self.net_window.append(net_input)
        self.state_window.pop(0)
        self.state_window.append(input_array)
        self.action_window.pop(0)
        self.action_window.append(action)

    def backward(self, reward):
        self.latest_reward = reward
        self.average_reward_window.add(reward)
        self.reward_window.pop(0)
        self.reward_window.append(reward)

        if not self.learning: 
            return

        self.age += 1

        #it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
        #(given that an appropriate number of state measurements already exist, of course)
        if self.forward_passes > self.temporal_window + 1:
            n = self.window_size
            e = Experience(
                self.net_window[n - 2],
                self.action_window[n - 2],
                self.reward_window[n - 2],
                self.net_window[n - 1]
            )

            if len(self.experience) < self.experience_size:
                self.experience.append(e)
            else:
                ri = randi(0, self.experience_size)
                self.experience[ri] = e

        #learn based on experience, once we have some samples to go on
        #this is where the magic happens...
        if len(self.experience) > self.start_learn_threshold:
            avcost = 0.0

            for k in xrange(self.tdtrainer.batch_size):
                re = randi(0, len(self.experience))
                e = self.experience[re]
                x = Vol(1, 1, self.net_inputs)
                x.w = e.state0
                maxact = self.policy(e.state1)
                r = e.reward0 + self.gamma * maxact.value
                ystruct = {'dim': e.action0, 'val': r}
                stats = self.tdtrainer.train(x, ystruct)
                avcost += stats['loss']

            avcost /= self.tdtrainer.batch_size
            self.average_loss_window.add(avcost)