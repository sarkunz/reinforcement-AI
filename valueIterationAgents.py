# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
         # Vk+1(s) <- max_a sum_s' T(s,a,s')[R(s,a,s') + g*V_k(s')]
        for it in range(self.iterations):
            stateVals = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state): 
                    continue

                qvals = [self.computeQValueFromValues(state, act) for act in self.mdp.getPossibleActions(state)]
                stateVals[state] = max(qvals)

            for state in self.mdp.getStates(): # need this??
                self.values[state] = stateVals[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gV*(s')] 
        qval = 0
        for n_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += prob * (self.mdp.getReward(state, action, n_state) + (self.discount * self.getValue(n_state)))
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #calc all qvals for state + return max action
        maxAct = None
        maxQ = -math.inf
        for act in self.mdp.getPossibleActions(state):
            qval = self.computeQValueFromValues(state, act)
            if qval > maxQ:
                maxQ = qval
                maxAct = act
        return maxAct

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for it in range(self.iterations):
            state = states[it % len(states)] #one state per iteration
            if self.mdp.isTerminal(state): continue
            qvals = [self.computeQValueFromValues(state, act) for act in self.mdp.getPossibleActions(state)]
            self.values[state] = max(qvals)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

        """
        Compute predecessors of all states.
        Initialize an empty priority queue.
        For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate over states in the order returned by self.mdp.getStates())
            Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
            Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        For iteration in 0, 1, 2, ..., self.iterations - 1, do:
            If the priority queue is empty, then terminate.
            Pop a state s off the priority queue.
            Update the value of s (if it is not a terminal state) in self.values.
            For each predecessor p of s, do:
                Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
                If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority. As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        """
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #compute predecessors of all states
        predecessors = {}
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state): 
                continue
            #get predecessors
            for act in self.mdp.getPossibleActions(state):
                for n_state, _ in self.mdp.getTransitionStatesAndProbs(state, act):
                    if n_state not in predecessors:
                        predecessors[n_state] = set()
                    predecessors[n_state].add(state)
            #find the abs val of the dif between cur val s in self.vals and highest q valaccross all possible action s s
            maxq = max([self.computeQValueFromValues(state, act) for act in self.mdp.getPossibleActions(state)])
            diff = abs(self.values[state] - maxq)
            pq.update(state, -diff)

        for it in range(self.iterations):
            if pq.isEmpty(): break
            state = pq.pop()
            self.values[state] = max([self.computeQValueFromValues(state, act) for act in self.mdp.getPossibleActions(state)])
            for pred in predecessors[state]:
                #calc diff between pred value and maxq
                maxq = max([self.computeQValueFromValues(pred, act) for act in self.mdp.getPossibleActions(pred)])
                diff = abs(self.values[pred] - maxq)
                if diff > self.theta:
                    pq.update(pred, -diff)

