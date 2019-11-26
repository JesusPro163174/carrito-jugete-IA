#esta clase creara la red neuronal dinamicamente
import numpy as np
import activationFunctions
class NeuronalNetwork:
    #layers = numero de capas que tendra nuestra red
    #activation = sera la  funcion de activacion (tanh,sigm)
    def __init__(self,capas,function_activacion):

        self.capas = capas
        
        if function_activacion ==  'tanh':
            self.function_activacion = tanh
            self.function_activacion_primaria = tanh_derivada
        else: 
            self.function_activacion            = sigmoid
            self.function_activacion_primaria   = sigmoid_derivada
        
        #arreglos de pesos y deltas
        self.pesos  = []
        self.deltas = []

        # numeros aleatorios de pesos, valores entre -1 a 1
        # se asignan valores aleatorios a capa de entrada y capa oculta
        for i in range(1, len(self.capas) - 1):
            r = 2*np.random.random((self.capas[i-1] + 1, self.capas[i] + 1)) -1
            self.pesos.append(r)
        # se asigna valores aleatorios a capa de salida
        r = 2*np.random.random( (self.capas[i] + 1, self.capas[i+1])) - 1
        self.pesos.append(r)

        self.print_pesos()
    
    def predict(self, x): 
        ones = np.atleast_2d(np.ones(x.shape[0])) #convierte un vertor a 2 dimenciones ones llena un vector con unos
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.function_activacion(np.dot(a, self.weights[l])) #dot hace el producto de dos array's
        return a

    def print_pesos(self):
        print("Pesos entre conexiones: ")
        for i in range(len(self.pesos)):
            print(self.pesos[i])

    def get_pesos(self):
        return self.pesos
    
    def get_deltas(self):
        return self.deltas
    




#functiones de activación
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivada(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1.0 - x**2

nn = NeuronalNetwork([1,2,3],'tanh')
        
