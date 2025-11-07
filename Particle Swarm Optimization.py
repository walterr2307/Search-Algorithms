from random import randint, uniform
from pandas import read_parquet, to_numeric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np


def somar(parcela1, parcela2, tipo_algoritmo):
    return [
        (
            round(parcela1[i] + parcela2[i], 4)
            if tipo_algoritmo > 0 and i == 1
            else parcela1[i] + parcela2[i]
        )
        for i in range(len(parcela1))
    ]


def subtrair(minuendo, subtraendo):
    return [minuendo[i] - subtraendo[i] for i in range(len(minuendo))]


def multiplicar(vetor, c, tipo_algoritmo):
    return [
        vetor[i] * c if tipo_algoritmo > 0 and i == 1 else round(vetor[i] * c)
        for i in range(len(vetor))
    ]


class Particula:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.perf = None
        self.c1 = uniform(0, 2)
        self.c2 = uniform(0, 2)
        self.melhor_pos = None
        self.melhor_perf = None

    def __str__(self):
        return "Position: {} --> Speed: {} --> Accuracy: {:.1f}%".format(
            self.pos, self.vel, self.perf
        )

    def prox_vel(self, melhor_pos_geral, tipo_algoritmo):
        self.vel = somar(
            multiplicar(subtrair(self.melhor_pos, self.pos), self.c1, tipo_algoritmo),
            multiplicar(subtrair(melhor_pos_geral, self.pos), self.c2, tipo_algoritmo),
            tipo_algoritmo,
        )
        self.vel = somar(self.vel, self.vel, tipo_algoritmo)


class PSO:
    def __init__(self, tam_enxame, num_interacoes, tipo_algoritmo, endereco_csv):
        self.tam_enxame = tam_enxame
        self.num_interacoes = num_interacoes
        self.tipo_algoritmo = tipo_algoritmo
        self.dataframe = read_parquet(endereco_csv)

        # Converte colunas datetime para int64
        datetime_cols = self.dataframe.select_dtypes(
            include=["datetime64[ns]", "datetime64[ns, UTC]"]
        ).columns
        for col in datetime_cols:
            self.dataframe[col] = self.dataframe[col].astype("int64") // 10**9

        # Converte todas as colunas para float para evitar erros no scikit-learn
        self.dataframe = (
            self.dataframe.sample(10000).apply(to_numeric, errors="coerce").fillna(0)
        )
        self.melhor_pos_geral = None
        self.melhor_perf_geral = 0
        self.weights_map = {0: "uniform", 1: "distance"}
        self.criterion_map = {0: "entropy", 1: "gini", 2: "log_loss"}
        self.kernel_map = {0: "sigmoid", 1: "poly", 2: "rbf", 3: "linear"}
        self.min = [
            [1, 0],
            [3, 0.1, 0],
            [3, 0.1, 0, 3],
            [0, 0.1],
            [50, 0.01, 1],
            [0.01],
        ]
        self.max = [
            [20, 1],
            [20, 0.3, 2],
            [20, 0.3, 2, 200],
            [3, 1],
            [200, 1.0, 10],
            [100],
        ]

        self.executar()

    def definir_xy(self):
        x = self.dataframe.iloc[:, :-1].values.astype(float)
        y = self.dataframe.iloc[:, -1].values
        self.x_treinamento, self.x_teste, self.y_treinamento, self.y_teste = (
            train_test_split(x, y, test_size=0.3, random_state=0)
        )

    # Método para gerar uma posição inicial para as partículas
    def pos_zero(self):
        zero = []
        for i in range(len(self.max[self.tipo_algoritmo])):
            zero.append(self.max[self.tipo_algoritmo][i] // 2)
        return zero

    # Método para gerar uma posição aleatória para as partículas de acordo com o tipo de algoritmo
    def pos_aleatoria(self):
        if self.tipo_algoritmo == 0:  # KNN
            return [randint(1, 20), randint(0, 1)]
        elif self.tipo_algoritmo == 1:  # DecisionTree
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2)]
        elif self.tipo_algoritmo == 2:  # RandomForest
            return [
                randint(3, 20),
                round(uniform(0.1, 0.3), 4),
                randint(0, 2),
                randint(3, 200),
            ]
        elif self.tipo_algoritmo == 3:  # SVC
            return [randint(0, 3), round(uniform(0.1, 1), 4)]
        elif self.tipo_algoritmo == 4:  # Gradient Boosting
            return [randint(50, 200), round(uniform(0.01, 1.0), 3), randint(1, 10)]
        elif self.tipo_algoritmo == 5:  # Logistic Regression
            return [round(uniform(0.01, 100), 3)]

    # Método para gerar uma partícula com posição e velocidade aleatórias
    def gerar_particula(self):
        pos = self.pos_aleatoria()
        vel = subtrair(pos, self.pos_zero())
        p = Particula(pos, vel)
        p.melhor_pos = list(p.pos)
        p.perf = self.fitness(p)
        p.melhor_perf = p.perf
        return p

    # Método para gerar um modelo de acordo com a posição da partícula
    def gerar_modelo(self, p):
        if self.tipo_algoritmo == 0:  # KNN
            return KNeighborsClassifier(
                n_neighbors=p.pos[0], weights=self.weights_map[p.pos[1]]
            )
        elif self.tipo_algoritmo == 1:  # Decision Tree
            return DecisionTreeClassifier(
                max_depth=p.pos[0],
                min_samples_split=p.pos[1],
                criterion=self.criterion_map[p.pos[2]],
            )
        elif self.tipo_algoritmo == 2:  # Random Forest
            return RandomForestClassifier(
                max_depth=p.pos[0],
                min_samples_split=p.pos[1],
                criterion=self.criterion_map[p.pos[2]],
                n_estimators=p.pos[3],
            )
        elif self.tipo_algoritmo == 3:  # SVC
            return SVC(kernel=self.kernel_map[p.pos[0]], C=p.pos[1])
        elif self.tipo_algoritmo == 4:  # Gradient Boosting
            return GradientBoostingClassifier(
                n_estimators=p.pos[0], learning_rate=p.pos[1], max_depth=p.pos[2]
            )
        elif self.tipo_algoritmo == 5:  # Logistic Regression
            return LogisticRegression(C=p.pos[0], max_iter=500, solver="lbfgs")

    # Método para calcular o desempenho de um modelo com base na posição da partícula
    def fitness(self, p):
        modelo = self.gerar_modelo(p)
        modelo.fit(self.x_treinamento, self.y_treinamento)
        previsoes = modelo.predict(self.x_teste)
        return accuracy_score(self.y_teste, previsoes) * 100

    # Método para gerar o enxame inicial
    def gerar_enxame(self):
        enxame = []
        for i in range(self.tam_enxame):
            p = self.gerar_particula()
            enxame.append(p)
            if p.melhor_perf > self.melhor_perf_geral:
                self.melhor_perf_geral = p.melhor_perf
                self.melhor_pos_geral = list(p.melhor_pos)
        return enxame

    # Método para mover as partículas e atualizar suas posições e velocidades
    def mover(self, p):
        p.prox_vel(self.melhor_pos_geral, self.tipo_algoritmo)
        p.pos = somar(p.pos, p.vel, self.tipo_algoritmo)

        for i in range(len(self.max[self.tipo_algoritmo])):
            if p.pos[i] > self.max[self.tipo_algoritmo][i]:
                p.pos[i] = self.max[self.tipo_algoritmo][i]
            if p.pos[i] < self.min[self.tipo_algoritmo][i]:
                p.pos[i] = self.min[self.tipo_algoritmo][i]

        p.perf = self.fitness(p)

        if p.perf > p.melhor_perf:
            p.melhor_perf = p.perf
            p.melhor_pos = list(p.pos)
        if p.melhor_perf > self.melhor_perf_geral:
            self.melhor_perf_geral = p.melhor_perf
            self.melhor_pos_geral = list(p.melhor_pos)

    # Método principal que executa o algoritmo PSO
    def executar(self):
        self.definir_xy()
        enxame = self.gerar_enxame()

        with open("PSO.txt", "w") as file:
            file.write("1* Interation:\n")

            for i in range(self.tam_enxame):
                file.write(str(enxame[i]) + "\n")

            for j in range(1, self.num_interacoes):
                file.write("\n{}* Interation:\n".format(j + 1))

                for i in range(self.tam_enxame):
                    self.mover(enxame[i])
                    file.write(str(enxame[i]) + "\n")

            file.write(
                "\nBest Position: {}\nBest Accuracy: {:.1f}%".format(
                    self.melhor_pos_geral, self.melhor_perf_geral
                )
            )


# Criando uma instância da classe PSO com os parâmetros específicos
pso = PSO(10, 10, 3, "Dataset_com_temperatura_final.parquet")
