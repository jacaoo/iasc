# Primeira camada, onde entram os dados, isto vai servir para preparar o ambiente, o ds é dimensão de entrada dos valores
class InputLayer:
    def __init__(self, ds):
        # Cria um array de 0s
        self.y = [0 for _ in range(ds)]
        self.ds = ds

    def propagate(self, x):
        self.y = x
        return self.y