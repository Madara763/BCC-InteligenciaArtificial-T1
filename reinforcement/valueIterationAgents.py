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
        """
        Executa o algoritmo de iteração de valor pelo número de iterações definido.
        """
        for i in range(self.iterations):
            # Usamos um dicionário temporário porque a atualização deve ser em lote (batch)
            novos_valores = util.Counter()
            estados = self.mdp.getStates()
            
            for estado in estados:
                # Estados terminais sempre têm valor 0
                if self.mdp.isTerminal(estado):
                    continue
                
                acoes_legais = self.mdp.getPossibleActions(estado)
                # O novo valor do estado é o Q-Value máximo entre as ações possíveis
                melhor_valor = max([self.computeQValueFromValues(estado, acao) for acao in acoes_legais])
                novos_valores[estado] = melhor_valor
                
            # Ao final do ciclo de todos os estados, atualizamos a tabela oficial
            self.values = novos_valores


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
        """
        Calcula o Q-Value (Q(s,a)) a partir dos valores atuais (V(s')).
        """
        q_valor = 0.0
        transicoes = self.mdp.getTransitionStatesAndProbs(state, action)
        
        for proximo_estado, probabilidade in transicoes:
            recompensa = self.mdp.getReward(state, action, proximo_estado)
            desconto = self.discount
            valor_futuro = self.values[proximo_estado]
            
            # Equação de Bellman: soma de P(s') * [R + gama * V(s')]
            q_valor += probabilidade * (recompensa + desconto * valor_futuro)
            
        return q_valor

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        """
        Retorna a melhor ação (política ótima atual) para o estado dado.
        """
        if self.mdp.isTerminal(state):
            return None
            
        acoes_legais = self.mdp.getPossibleActions(state)
        melhor_acao = None
        maior_q_valor = -float('inf')
        
        for acao in acoes_legais:
            q_valor = self.computeQValueFromValues(state, acao)
            if q_valor > maior_q_valor:
                maior_q_valor = q_valor
                melhor_acao = acao
                
        return melhor_acao

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

