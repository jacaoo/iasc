import math
from NeuralNetwork import NeuralNetwork

# Ficheiro para testes da rede neuronal

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def tanh_derivada(x):
    t = tanh(x)
    return 1 - (t ** 2)


def main():
    X = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1]
    ]

    Y = [
        [0],
        [1],
        [1],
        [1],
        [0],
        [0],
        [0]
    ]

    my_network = NeuralNetwork([9, 3, 1], [tanh, tanh_derivada])

    first_guess = my_network.propagate(X[1])
    print(f"Palpite ANTES do treino (Deveria ser perto de 1): {first_guess}")


    data = my_network.train(X, Y, 1000, 0.05, 0.2, 0.1)

    print(data)

    for i in range(len(X)):
        new_guess = my_network.propagate(X[i])
        print(f"Palpite DEPOIS do treino : {new_guess}")

main()