# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, estado_atual: GameState, acao):
        # Extrai informações úteis do estado sucessor
        estado_sucessor = estado_atual.generatePacmanSuccessor(acao)
        nova_posicao = estado_sucessor.getPacmanPosition()
        nova_comida = estado_sucessor.getFood()
        novos_estados_fantasmas = estado_sucessor.getGhostStates()
        
        # 1. Pontuação base do jogo
        pontuacao = estado_sucessor.getScore()

        # 2. Lógica da Comida
        lista_comidas = nova_comida.asList()
        if len(lista_comidas) > 0:
            from util import manhattanDistance #
            distancia_minima_comida = min([manhattanDistance(nova_posicao, comida) for comida in lista_comidas])
            # Dá um peso de 10 ao inverso da distancia
            pontuacao += 10.0 / distancia_minima_comida 

        # 3. Lógica dos Fantasmas
        for fantasma in novos_estados_fantasmas:
            posicao_fantasma = fantasma.getPosition()
            from util import manhattanDistance
            distancia_fantasma = manhattanDistance(nova_posicao, posicao_fantasma)
            
        if distancia_fantasma <= 1:
            pontuacao -= 99999

        # 4. Não ficar parado
        from game import Directions 
        if acao == Directions.STOP:
            pontuacao -= 50

        return pontuacao

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        # Função recursiva auxiliar
        def minimax(agente, profundidade, estado):
            # 1. Condições de Paragem (Base Cases)
            # Pára se o Pac-Man ganhou, perdeu, ou se atingiu a profundidade maxima
            if estado.isWin() or estado.isLose() or profundidade == self.depth:
                return self.evaluationFunction(estado)

            # 2. Turno do Pac-Man (Maximizador - Agente 0)
            if agente == 0:
                maior_valor = -float('inf')
                for acao in estado.getLegalActions(agente):
                    sucessor = estado.generateSuccessor(agente, acao)
                    # O próximo a jogar é o Fantasma 1 (agente = 1), na mesma profundidade
                    valor = minimax(1, profundidade, sucessor)
                    if valor > maior_valor:
                        maior_valor = valor
                return maior_valor

            # 3. Turno dos Fantasmas (Minimizadores - Agente > 0)
            else:
                menor_valor = float('inf')
                proximo_agente = agente + 1
                proxima_profundidade = profundidade

                # LÓGICA CHAVE: Se este for o último fantasma, o próximo a jogar é 
                # o Pac-Man (0) e a profundidade da árvore aumenta em 1!
                if proximo_agente == estado.getNumAgents():
                    proximo_agente = 0
                    proxima_profundidade += 1

                for acao in estado.getLegalActions(agente):
                    sucessor = estado.generateSuccessor(agente, acao)
                    valor = minimax(proximo_agente, proxima_profundidade, sucessor)
                    if valor < menor_valor:
                        menor_valor = valor
                return menor_valor

        # --- Raiz da Árvore de Busca (dentro de getAction) ---
        from game import Directions
        melhor_acao = Directions.STOP
        maior_valor_encontrado = -float('inf')

        # O Pac-Man avalia todas as suas ações possíveis no estado atual
        for acao in gameState.getLegalActions(0):
            sucessor = gameState.generateSuccessor(0, acao)
            # O valor desta ação é o resultado do minimax a partir do Fantasma 1 (profundidade 0)
            valor = minimax(1, 0, sucessor)
            
            # Queremos a ação que leve ao maior valor possível
            if valor > maior_valor_encontrado:
                maior_valor_encontrado = valor
                melhor_acao = acao

        return melhor_acao

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Devolve a ação minimax usando self.depth e self.evaluationFunction,
        mas agora com poda Alpha-Beta.
        """

        # Função recursiva auxiliar agora recebe alpha e beta
        def alphabeta(agente, profundidade, estado, alpha, beta):
            # 1. Condições de Paragem (iguais às do Minimax)
            if estado.isWin() or estado.isLose() or profundidade == self.depth:
                return self.evaluationFunction(estado)

            # 2. Turno do Pac-Man (MAX)
            if agente == 0:
                valor = -float('inf')
                for acao in estado.getLegalActions(agente):
                    sucessor = estado.generateSuccessor(agente, acao)
                    valor = max(valor, alphabeta(1, profundidade, sucessor, alpha, beta))
                    
                    # PODA ALPHA-BETA PARA O MAX:
                    # Se o valor encontrado for MAIOR que o beta que os fantasmas já têm,
                    # os fantasmas nunca vão deixar o Pac-Man chegar aqui. Poda!
                    if valor > beta:
                        return valor
                    
                    # Atualiza o alpha com o melhor valor encontrado até agora
                    alpha = max(alpha, valor)
                return valor

            # 3. Turno dos Fantasmas (MIN)
            else:
                valor = float('inf')
                proximo_agente = agente + 1
                proxima_profundidade = profundidade

                if proximo_agente == estado.getNumAgents():
                    proximo_agente = 0
                    proxima_profundidade += 1

                for acao in estado.getLegalActions(agente):
                    sucessor = estado.generateSuccessor(agente, acao)
                    valor = min(valor, alphabeta(proximo_agente, proxima_profundidade, sucessor, alpha, beta))
                    
                    # PODA ALPHA-BETA PARA O MIN:
                    # Se o valor encontrado for MENOR que o alpha que o Pac-Man já tem,
                    # o Pac-Man nunca vai escolher este caminho. Poda!
                    if valor < alpha:
                        return valor
                        
                    # Atualiza o beta com o pior valor (para o Pac-Man) encontrado até agora
                    beta = min(beta, valor)
                return valor

        # --- Raiz da Árvore de Busca (Onde a mágica começa) ---
        from game import Directions
        melhor_acao = Directions.STOP
        maior_valor = -float('inf')
        
        # Inicializamos alpha e beta
        alpha = -float('inf')
        beta = float('inf')

        # O ciclo inicial também atua como um nó MAX (turno do Pac-Man)
        for acao in gameState.getLegalActions(0):
            sucessor = gameState.generateSuccessor(0, acao)
            valor = alphabeta(1, 0, sucessor, alpha, beta)
            
            if valor > maior_valor:
                maior_valor = valor
                melhor_acao = acao
                
            # Na raiz, também temos de atualizar o alpha após cada ramo avaliado!
            alpha = max(alpha, maior_valor)

        return melhor_acao

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Devolve a ação expectimax usando self.depth e self.evaluationFunction.
        Os fantasmas são modelados como escolhendo uniformemente ao acaso.
        """

        # Função recursiva auxiliar
        def expectimax(agente, profundidade, estado):
            # 1. Condições de Paragem
            if estado.isWin() or estado.isLose() or profundidade == self.depth:
                return self.evaluationFunction(estado)

            # 2. Turno do Pac-Man (MAX) - Igual ao Minimax!
            if agente == 0:
                maior_valor = -float('inf')
                for acao in estado.getLegalActions(agente):
                    sucessor = estado.generateSuccessor(agente, acao)
                    valor = expectimax(1, profundidade, sucessor)
                    if valor > maior_valor:
                        maior_valor = valor
                return maior_valor

            # 3. Turno dos Fantasmas (EXPECT / CHANCE)
            else:
                valor_esperado = 0.0
                acoes_legais = estado.getLegalActions(agente)
                
                # A probabilidade de cada ação é igual para todas
                probabilidade = 1.0 / len(acoes_legais)
                
                proximo_agente = agente + 1
                proxima_profundidade = profundidade

                # Lógica de transição de turno (se for o último fantasma, volta ao Pac-Man e desce na árvore)
                if proximo_agente == estado.getNumAgents():
                    proximo_agente = 0
                    proxima_profundidade += 1

                for acao in acoes_legais:
                    sucessor = estado.generateSuccessor(agente, acao)
                    valor = expectimax(proximo_agente, proxima_profundidade, sucessor)
                    
                    # Em vez de min(), somamos o valor proporcional à sua probabilidade
                    valor_esperado += (valor * probabilidade)
                    
                return valor_esperado

        # --- Raiz da Árvore de Busca ---
        from game import Directions
        melhor_acao = Directions.STOP
        maior_valor_encontrado = -float('inf')

        # O Pac-Man (MAX) avalia as suas opções iniciais
        for acao in gameState.getLegalActions(0):
            sucessor = gameState.generateSuccessor(0, acao)
            valor = expectimax(1, 0, sucessor)
            
            if valor > maior_valor_encontrado:
                maior_valor_encontrado = valor
                melhor_acao = acao

        return melhor_acao

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    from util import manhattanDistance #
    
    # 1. Casos Extremos: Vitória ou Derrota
    if currentGameState.isWin():
        return float('inf') #
    if currentGameState.isLose():
        return -float('inf') #

    # Extrair as informações do estado
    posicao_pacman = currentGameState.getPacmanPosition()
    lista_comidas = currentGameState.getFood().asList()
    estados_fantasmas = currentGameState.getGhostStates()
    lista_capsulas = currentGameState.getCapsules()

    # Começamos com a pontuação do jogo
    pontuacao = currentGameState.getScore()

    # 2. Lógica da Comida (Mais perto = Melhor, Menos comida restante = Melhor)
    if len(lista_comidas) > 0:
        distancia_minima_comida = min([manhattanDistance(posicao_pacman, comida) for comida in lista_comidas])
        pontuacao += 10.0 / distancia_minima_comida
        
    # Penaliza fortemente a quantidade de comida e cápsulas que ainda sobram no mapa
    pontuacao -= 20.0 * len(lista_comidas)
    pontuacao -= 20.0 * len(lista_capsulas)

    # 3. Lógica dos Fantasmas
    for fantasma in estados_fantasmas:
        distancia_fantasma = manhattanDistance(posicao_pacman, fantasma.getPosition())
        
        if fantasma.scaredTimer > 0:
            # Fantasma assustado: O Pac-Man vira um caçador!
            pontuacao += 100.0 / (distancia_fantasma + 1)
        else:
            # Fantasma ativo: Risco de vida
            if distancia_fantasma <= 1:
                pontuacao -= 99999

    return pontuacao

# Abbreviation
better = betterEvaluationFunction
