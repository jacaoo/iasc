from Neuron import Neuron

# Isto vai ser as camadas entre-médias da rede neuronal
# de - quantidade de ligações vai receber
# ds - quantidade de saídas
class DenseLayer:
    def __init__(self, de, ds, sigma):
        self.neurons = [Neuron(de, sigma) for _ in range(ds)]
        self.de = de
        self.ds = ds
        self.sigma = sigma

    # Retorna a saída do neurónio
    @property
    def y(self):
        return [neuron.y for neuron in self.neurons]

    # Fazer basicamente o cálculo do peso, b e função de ativação conforme feito em cada neurónio
    def propagate(self, x):
        y = [neuron.propagate(x) for neuron in self.neurons]
        return y

    # o y é a camada anterior
    def adapt(self, error_prop, y, a, beta):
        for j in range(0, self.ds):
            self.neurons[j].adapt(error_prop[j], y, a, beta)