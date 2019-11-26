#esta clase creara la red neuronal dinamicamente
import numpy as np
import activationFunctions
import matplotlib.pyplot as plt
class NeuronalNetwork:
    #capas      = numero de capas que tendra nuestra red
    #activacion = sera la  funcion de activacion (tanh,sigm)
    def __init__(self,capas,function_activacion):

        self.capas = capas
        
        self.function_activacion = tanh
        self.function_activacion_primaria = tanh_derivada
        
        #arreglos de pesos y deltas
        self.pesos  = []
        self.deltas = []
        self.errores = []
        self.epoca  = []

        # numeros aleatorios de pesos, valores entre -1 a 1
        # se asignan valores aleatorios a capa de entrada y capa oculta
        for i in range(1, len(self.capas) - 1):
            r = 2*np.random.random((self.capas[i-1] + 1, self.capas[i] + 1)) -1
            self.pesos.append(r)
        # se asigna valores aleatorios a capa de salida
        r = 2*np.random.random((self.capas[i] + 1, self.capas[i+1]))-1
        self.pesos.append(r)

        #self.print_pesos()
    
    def ajuste(self,X,y,tasa_aprendizaje=0.02,epocas=100000):
        # Agrego columna de unos a las entradas X
        # Con esto agregamos la unidad de Bias a la capa de entrada
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epocas):
            
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            #forward pass
            for l in range(len(self.pesos)):
                dot_value = np.dot(a[l], self.pesos[l])
                activacion = self.function_activacion(dot_value)
                a.append(activacion)
            # Calculo la diferencia en la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.function_activacion_primaria(a[-1])]
            
            # Empezamos en la segunda capa hasta el ultimo
            # (Una capa anterior a la de salida)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.pesos[l].T)*self.function_activacion_primaria(a[l]))
            self.deltas.append(deltas)

            # invertir
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()
            #backward pass
            # backpropagation
            # 1. Multiplcar los delta de salida con las activaciones de entrada 
            #    para obtener el gradiente del peso.
            # 2. actualizo el peso restandole un porcentaje del gradiente
            for i in range(len(self.pesos)):
                capas = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.pesos[i] += tasa_aprendizaje * capas.T.dot(delta)

            if k % 10000 == 0: self.aprendizaje(error,k)

    def predict(self, x): 
        ones = np.atleast_2d(np.ones(x.shape[0])) #convierte un vertor a 2 dimenciones ones llena un vector con unos
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.pesos)):
            a = self.function_activacion(np.dot(a, self.pesos[l])) #dot hace el producto de dos array's
        return a

    def print_pesos(self):
        print("Pesos entre conexiones: ")
        for i in range(len(self.pesos)):
            print(self.pesos[i])

    def get_pesos(self):
        return self.pesos
    
    def get_deltas(self):
        return self.deltas
    
    def aprendizaje(self,error,epoca):
        print("Error: ",error)
        print("epoca: ",epoca)
        
    
#function de activación
def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1.0 - x**2

def valNN(x):
    return (int)(abs(round(x)))

capas = [2,3,4]
nn = NeuronalNetwork(capas,'tanh')
X = np.array([
    [-1,0],   # sin obstaculos
    [-1,1],   # sin obstaculos
    [-1,-1],  # sin obstaculos
    [0,-1],   # obstaculo detectado a derecha
    [0,1],    # obstaculo a izq
    [0,0],    # obstaculo centro
    [1,1],    # demasiado cerca a derecha
    [1,-1],   # demasiado cerca a izq
    [1,0]     # demasiado cerca centro
])
# las salidas 'y' se corresponden con encender (o no) los motores
y = np.array([
    [1,0,0,1], # avanzar
    [1,0,0,1], # avanzar
    [1,0,0,1], # avanzar
    [0,1,0,1], # giro derecha
    [1,0,1,0], # giro izquierda (cambie izq y derecha)
    [1,0,0,1], # avanzar
    [0,1,1,0], # retroceder
    [0,1,1,0], # retroceder
    [0,1,1,0]  # retroceder
])
nn.ajuste(X, y, tasa_aprendizaje=0.2,epocas=500000)

index=0
for e in X:
    prediccion = nn.predict(e)
    print("X:",e,"esperado:",y[index],"obtenido:", valNN(prediccion[0]),valNN(prediccion[1]),valNN(prediccion[2]),valNN(prediccion[3]))
    index=index+1


deltas = nn.get_deltas()
print(deltas)
valores=[]
index=0
for arreglo in deltas:
    valores.append(arreglo[1][0] + arreglo[1][1])
    index=index+1

plt.plot(range(len(valores)), valores, color='b')
plt.ylim([0, 0.4])
plt.ylabel('Cost')
plt.xlabel('Epocas')
plt.tight_layout()
plt.show()
        
