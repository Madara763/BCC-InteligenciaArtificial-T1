from torch.nn.functional import mse_loss, cross_entropy

def regression_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    Inputs:
        y_pred: a node with shape (batch_size x 1), containing the predicted y-values
        y: a node with shape (batch_size x 1), containing the true y-values
            to be used for training
    Returns: a tensor of size 1 containing the loss
    """
    "*** YOUR CODE HERE ***"
    # Calculamos o Erro Quadrático Médio entre as previsões do modelo (y_pred)
    # e os valores reais da função seno (y)
    erro_quadratico = mse_loss(y_pred, y)
    
    return erro_quadratico



def digitclassifier_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred: a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """
    # A função de perda para classificação multiclasse é a Entropia Cruzada
    entropia_cruzada = cross_entropy(y_pred, y)
    
    return entropia_cruzada


def languageid_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a node with shape
    (batch_size x 5). Each row is a one-hot vector encoding the correct
    language.

    Inputs:
        model: Pytorch model to use
        y_pred: a node with shape (batch_size x 5)
        y: a node with shape (batch_size x 5)
    Returns: a loss node
    """
    "*** YOUR CODE HERE ***"
    # Usamos Entropia Cruzada para prever o idioma correto entre as 5 opções
    entropia_cruzada = cross_entropy(y_pred, y)
    
    return entropia_cruzada


def digitconvolution_Loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred : a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """
    # Usamos a Entropia Cruzada, pois o objetivo final ainda é classificar os 10 dígitos
    entropia_cruzada = cross_entropy(y_pred, y)
    
    return entropia_cruzada
    
