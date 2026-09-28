import random

# Nota - Comentários em português, mas o código em inglês porque fica aesthetic (mais bonito)

# Classe para definir o que é o neurónio
# tem como entrada a função de ativação aka sigma
# e o d que vai servir para a indicação do volume de entrada
class Neuron:
    # Construtor da classe
    def __init__(self, d, sigma):
        # W - Vetor de pesos que vai começar com -1's e 1s depois com o treino isso muda
        self.w = [random.uniform(-1, 1) for _ in range(d)]
        # B - Valor que depois decide qual é o resultado suposto, ou seja, se for maior que b passa, neste momento fica -1 ou 1
        self.b = random.uniform(-1, 1)
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

    # Ajustar os pesos consuante o resultado da camada anterior
    # y_last - vetor de saída da camada anterior
    # a - valor que dita o quão o peso do atual neurónio é que têm de ser ajustado
    def adapt(self, error_prop, y_last, a):
        delta_w = [item * (-a * self.derivativeY * error_prop) for item in y_last]
        self.w = [w + delta_w for w, delta_w in zip(delta_w, self.w)]
        delta_b = -a * self.derivativeY * error_prop
        self.b = self.b + delta_b