import random

# Nota - Comentários em português, mas o código em inglês porque fica aesthetic (mais bonito)

# Classe para definir o que é o neurónio
class Neuron:
    # Construtor da classe
    # d - é a quantidade de entradas
    # sigma - função de ativação do neurónio
    def __init__(self, d, sigma):
        # W - Vetor de pesos (valor que define a importância de uma entrada específica) que vai começar com −1's e 1s depois com o treino isso muda
        self.w = [random.uniform(-1, 1) for _ in range(d)]
        # b - Valor que decide o valor mínimo da combinação de entradas e pesos para o neurónio ficar ativo
        self.b = random.uniform(-1, 1)
        # O valor de deltas é taxa de aprendizagem a ser adotada por cada w ou b respectivamente
        # Delta W - Vetor com a variação dos pesos associado a cada entrada
        self.delta_w = [0 for _ in range(d)]
        # Delta B - Valor de variação do bias
        self.delta_b = 0
        # H - Soma das ativações das entradas
        self.h = 0
        # y - Saída do neurónio
        self.y = 0
        # y' - Saída da primeira camada do neurónio
        self.derivativeY = 0
        self.d = d
        self.sigma = sigma

    # Esta função têm o propósito de receber o valor e aplicar o peso e depois a função de ativação e retornar o resultado
    def propagate(self, x):
        # Produto escalar entre a entrada e o seu peso associado
        dot_product = sum(peso * input_x for peso, input_x in zip(self.w, x))
        # A soma do resultado do produto escalar mais o b referido em cima para depois definir melhor a decisão
        self.h = dot_product + self.b
        # Aplicar a função de ativação ao resultado, é [0] porque o [1] contém a derivada
        self.y = self.sigma[0](self.h)
        self.derivativeY = self.sigma[1](self.h)
        return self.y

    # Ajustar os pesos consoante o resultado da camada anterior
    # y_last - vetor de saída da camada anterior
    # a - valor que dita o quão o peso do atual neurónio é que têm de ser ajustado
    def adapt(self, error_prop, y, a, beta):
        moment_delta = [beta * value for value in self.delta_w]
        self.delta_w = [item * (-a * self.derivativeY * error_prop) + moment_w for item, moment_w in zip(y, moment_delta)]
        self.w = [w + delta_w for w, delta_w in zip(self.delta_w, self.w)]
        moment_b = beta * self.delta_b
        self.delta_b = -a * self.derivativeY * error_prop + moment_b
        self.b = self.b + self.delta_b