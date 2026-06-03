from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Treina o perceptron até a convergência.
    """
    with no_grad():
        # Cria o carregador de dados (DataLoader)
        carregador_dados = DataLoader(dataset, batch_size=1, shuffle=True)
        
        convergiu = False
        
        # O treino continua até completarmos uma época (epoch) inteira sem erros
        while not convergiu:
            convergiu = True # Assumimos que convergiu até errar uma previsão
            
            for lote in carregador_dados:
                recursos_entrada = lote['x']
                rotulo_real = lote['label']
                
                # Pedimos ao modelo para tentar prever
                previsao = model.get_prediction(recursos_entrada)
                
                # Se a previsão falhar, o modelo não convergiu e os pesos precisam ser ajustados
                if previsao != rotulo_real.item():
                    convergiu = False
                    
                    # Regra do Perceptron: pesos_novos = pesos_atuais + (rotulo_real * recursos_entrada)
                    # Usamos model.get_weights() para modificar a variável diretamente
                    pesos_do_modelo = model.get_weights()
                    pesos_do_modelo += rotulo_real.item() * recursos_entrada
        


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"
    # Diminuímos a taxa de aprendizado para 0.005 para evitar oscilações finais
    otimizador = optim.Adam(model.parameters(), lr=0.005)
    
    # Aumentamos o tamanho do lote para estabilizar os gradientes
    tamanho_lote = 64
    carregador_dados = DataLoader(dataset, batch_size=tamanho_lote, shuffle=True)
    
    while True:
        perda_acumulada = 0.0
        total_lotes = 0
        
        for lote in carregador_dados:
            recursos_entrada = lote['x']
            rotulos_reais = lote['label']
            
            otimizador.zero_grad()
            
            previsoes = model(recursos_entrada)
            perda = regression_loss(previsoes, rotulos_reais)
            
            perda.backward()
            otimizador.step()
            
            perda_acumulada += perda.item()
            total_lotes += 1
            
        perda_media = perda_acumulada / total_lotes
        
        # Apertamos o critério de parada para ter margem de segurança no autograder
        if perda_media < 0.01:
            break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    
    # Otimizador Adam com taxa de aprendizado padrão (0.001) para maior estabilidade
    otimizador = optim.Adam(model.parameters(), lr=0.001)
    
    tamanho_lote = 64
    carregador_dados = DataLoader(dataset, batch_size=tamanho_lote, shuffle=True)
    
    # Treinamos até que a perda média seja baixa o suficiente para garantir os >97%
    while True:
        perda_acumulada = 0.0
        total_lotes = 0
        
        for lote in carregador_dados:
            recursos_entrada = lote['x']
            rotulos_reais = lote['label']
            
            otimizador.zero_grad()
            
            previsoes = model(recursos_entrada)
            perda = digitclassifier_loss(previsoes, rotulos_reais)
            
            perda.backward()
            otimizador.step()
            
            perda_acumulada += perda.item()
            total_lotes += 1
            
        perda_media = perda_acumulada / total_lotes
        
        # Paramos o treinamento quando a perda for inferior a 0.015
        if perda_media < 0.015:
            break


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"    
    # Otimizador com uma taxa de aprendizado ligeiramente maior para acelerar a RNN
    otimizador = optim.Adam(model.parameters(), lr=0.001)
    
    tamanho_lote = 64
    carregador_dados = DataLoader(dataset, batch_size=tamanho_lote, shuffle=True)
    
    while True:
        perda_acumulada = 0.0
        total_lotes = 0
        
        for lote in carregador_dados:
            recursos_entrada = lote['x']
            rotulos_reais = lote['label']
            
            # O dataloader retorna (tamanho_lote x tamanho_palavra x 47)
            # Trocamos a dimensão 0 pela 1 para ficar (tamanho_palavra x tamanho_lote x 47)
            recursos_reorganizados = movedim(recursos_entrada, 0, 1)
            
            # Convertendo para lista, cada item será um lote da i-ésima letra da palavra
            lista_letras = list(recursos_reorganizados)
            
            otimizador.zero_grad()
            
            previsoes = model(lista_letras)
            perda = languageid_loss(previsoes, rotulos_reais)
            
            perda.backward()
            otimizador.step()
            
            perda_acumulada += perda.item()
            total_lotes += 1
            
        perda_media = perda_acumulada / total_lotes
        
        # Limite rígido de paragem: garante a precisão de validação necessária (> 81%)
        if perda_media < 0.015:
            break



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
    model.train()
    
    # Otimizador Adam para ajustar os pesos da rede
    otimizador = optim.Adam(model.parameters(), lr=0.005)
    
    # O lote pode ser menor aqui porque o dataset do autograder (Dataset2) tem apenas 200 itens
    tamanho_lote = 32
    carregador_dados = DataLoader(dataset, batch_size=tamanho_lote, shuffle=True)
    
    while True:
        perda_acumulada = 0.0
        total_lotes = 0
        
        for lote in carregador_dados:
            recursos_entrada = lote['x']
            rotulos_reais = lote['label']
            
            otimizador.zero_grad()
            
            # Passa a imagem pela convolução e pelas camadas lineares
            previsoes = model(recursos_entrada)
            
            # Calcula o erro usando a função que construímos em losses.py
            perda = digitconvolution_Loss(previsoes, rotulos_reais)
            
            perda.backward()
            otimizador.step()
            
            perda_acumulada += perda.item()
            total_lotes += 1
            
        perda_media = perda_acumulada / total_lotes
        
        # Como o dataset é pequeno, podemos forçar um erro bem baixo para garantir os 80% de precisão
        if perda_media < 0.01:
            break
