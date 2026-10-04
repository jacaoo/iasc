from DenseLayer import DenseLayer
from InputLayer import InputLayer

# Isto vai ser a rede neuronal que envolve os neurónios e as camadas definidas com os mesmos
# shape - Vai ser um array com quantos neurónios vai ter por camada (cada index é uma camada)
class NeuralNetwork:
    # O init vai construir o formato da rede, as camadas e quantos neurónios por camada
    def __init__(self, shape, sigma):
        self.layers = []
        # N - número de camadas
        self.N = len(shape)
        ds1 = shape[0]  # primeira camada
        layer1 = InputLayer(ds1)
        self.layers.append(layer1)
        for n in range(1, self.N):
            # Número de entradas da camada n
            den = shape[n - 1]
            # Número de saídas da camada n
            dsn = shape[n]
            layer = DenseLayer(den, dsn, sigma)
            self.layers.append(layer)

    def propagate(self, x):
        y = x
        for layer in self.layers:
            y = layer.propagate(y)
        return y

    def predict(self, X):
        Y = [self.propagate(x) for x in range(X)]
        return Y

    # Compara o resultado atual da rede com o resultado esperado
    def delta_result(self, yN, y):
        return [yN[i] - y[i] for i in range(len(y))]

    # Isto basicamente anda para trás nas camadas para ajustar o peso para ter um melhor resultado
    def retropropagate(self, error_prop_N, a, beta):
        self.error_prop = error_prop_N
        for n in range((self.N - 1), 0, -1):
            y_N_1 = self.layers[n - 1].y
            d_N_1 = self.layers[n - 1].ds
            d_N = self.layers[n].ds
            neuron_N = self.layers[n].neurons
            error_prop_1 = [sum(neuron_N[j].w[i] * self.error_prop[j] * neuron_N[j].derivativeY for j in range(d_N)) for i in range(0, d_N_1)]
            self.layers[n].adapt(self.error_prop, y_N_1, a, beta)
            self.error_prop = error_prop_1

    # Esta função basicamente é o que faz tudo acontecer, recebe o valor inicial da rede, depois calcula o erro, depois chama a função que ajusta os valores chamando a função que ajusta as coisas em cada camada
    def adapt(self, x, y, a, beta):
        y_N = self.propagate(x)
        delta_N = self.delta_result(y_N, y)
        self.retropropagate(delta_N, a, beta)
        K = len(delta_N)
        self.avg_error = 1 / K * sum(pow(delta_N[k], 2) for k in range(K))
        return self.avg_error

    # Função para treinar os resultados com base no x e no y, sendo o x a raw data e o y os resultados esperados
    def train(self, X, Y, n_seasons, avg_error_max, a, beta):
        for i in range(n_seasons):
            error_per_season = 0
            for x, y in zip(X, Y):
                error_x = self.adapt(x, y, a, beta)
                error_per_season = max(error_per_season, error_x)
            if error_per_season <= avg_error_max:
                break