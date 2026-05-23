# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        # Tabela para armazenar os Q-Values conhecidos
        self.q_valores = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Se o estado/ação não foi explorado, retorna 0.0
        return self.q_valores[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        acoes_legais = self.getLegalActions(state)
        if not acoes_legais:
            return 0.0
            
        # O Valor (V) do estado é o maior Q-Value entre as ações legais
        return max([self.getQValue(state, acao) for acao in acoes_legais])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        acoes_legais = self.getLegalActions(state)
        if not acoes_legais:
            return None
            
        melhor_valor = self.computeValueFromQValues(state)
        # Se houver empates (mesmo Q-Value), pegamos todas as melhores ações
        melhores_acoes = [acao for acao in acoes_legais if self.getQValue(state, acao) == melhor_valor]
        
        # O projeto exige que o desempate seja aleatório
        return random.choice(melhores_acoes)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        """
        Implementa a exploração Epsilon-Greedy.
        """
        acoes_legais = self.getLegalActions(state)
        if not acoes_legais:
            return None
            
        probabilidade_exploracao = self.epsilon
        
        # flipCoin retorna True com probabilidade 'epsilon'
        if util.flipCoin(probabilidade_exploracao):
            # Age aleatoriamente (Exploração)
            return random.choice(acoes_legais)
        else:
            # Pega a melhor ação conhecida (Exploração da Política)
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        """
        A atualização principal do Q-Learning.
        """
        valor_antigo = self.getQValue(state, action)
        valor_futuro = self.computeValueFromQValues(nextState)
        
        # A "Amostra" observada agora
        amostra = reward + self.discount * valor_futuro
        
        # Fórmula de atualização iterativa usando a taxa de aprendizado (alpha)
        novo_valor = (1 - self.alpha) * valor_antigo + self.alpha * amostra
        
        # Atualiza a tabela
        self.q_valores[(state, action)] = novo_valor

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.pesos = util.Counter()

    def getWeights(self):
        return self.pesos

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        """
        O Q-Value agora é o Produto Escalar (Dot Product) entre as características e os pesos.
        """
        caracteristicas = self.featExtractor.getFeatures(state, action)
        q_valor = 0.0
        
        for chave, valor in caracteristicas.items():
            q_valor += self.pesos[chave] * valor
            
        return q_valor

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        """
        Atualiza todos os pesos com base na diferença (erro temporal).
        """
        valor_futuro = self.getValue(nextState)
        valor_estimado_atual = self.getQValue(state, action)
        
        # Diferença = (Recompensa + Gama * Max_Q) - Q_Atual
        diferenca = (reward + self.discount * valor_futuro) - valor_estimado_atual
        
        caracteristicas = self.featExtractor.getFeatures(state, action)
        
        # Atualiza os pesos: w = w + alpha * diferença * f(s,a)
        for chave, valor in caracteristicas.items():
            self.pesos[chave] += self.alpha * diferenca * valor

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
