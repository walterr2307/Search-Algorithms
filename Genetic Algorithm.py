from random import randint, uniform
from pandas import read_parquet, to_numeric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class GeneticAlgorithm:
    def __init__(self, num_individuos, num_populacoes, chance_de_mutar, tipo_algoritmo, endereco_parquet):
        self.num_individuos = num_individuos
        self.num_populacoes = num_populacoes
        self.chance_de_mutar = chance_de_mutar
        self.tipo_algoritmo = tipo_algoritmo
        self.dataframe = read_parquet(endereco_parquet)

        # Converte colunas datetime para int64
        datetime_cols = self.dataframe.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            self.dataframe[col] = self.dataframe[col].astype("int64") // 10**9

        # Converte todas as colunas para float para evitar erros no scikit-learn
        self.dataframe = (
            self.dataframe.sample(10000).apply(to_numeric, errors="coerce").fillna(0)
        )

        self.melhor_individuo = None
        self.melhor_acuracia = 0
        self.weights_map = {0: "uniform", 1: "distance"}
        self.criterion_map = {0: "entropy", 1: "gini", 2: "log_loss"}
        self.kernel_map = {0: "sigmoid", 1: "poly", 2: "rbf", 3: "linear"}

        self.executar()

    def definir_xy(self):
        x = self.dataframe.iloc[:, :-1].values
        y = self.dataframe.iloc[:, -1].values
        self.x_treinamento, self.x_teste, self.y_treinamento, self.y_teste = train_test_split(
            x, y, test_size=0.25, random_state=0
        )

    def gerar_individuo(self):
        if self.tipo_algoritmo == 0:  # KNN
            return [randint(1, 20), randint(0, 1)]
        elif self.tipo_algoritmo == 1:  # Decision Tree
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2)]
        elif self.tipo_algoritmo == 2:  # Random Forest
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2), randint(3, 200)]
        elif self.tipo_algoritmo == 3:  # SVC
            return [randint(0, 3), round(uniform(0.1, 1), 4)]
        elif self.tipo_algoritmo == 4:  # Gradient Boosting
            return [randint(50, 200), round(uniform(0.01, 1.0), 3), randint(1, 10)]
        elif self.tipo_algoritmo == 5:  # Logistic Regression
            return [round(uniform(0.01, 100), 3)]

    def gerar_populacao(self):
        return [self.gerar_individuo() for _ in range(self.num_individuos)]

    def gerar_modelo(self, individuo):
        if self.tipo_algoritmo == 0:
            return KNeighborsClassifier(n_neighbors=individuo[0], weights=self.weights_map[individuo[1]])
        elif self.tipo_algoritmo == 1:
            return DecisionTreeClassifier(max_depth=individuo[0], min_samples_split=individuo[1],
                                          criterion=self.criterion_map[individuo[2]])
        elif self.tipo_algoritmo == 2:
            return RandomForestClassifier(max_depth=individuo[0], min_samples_split=individuo[1],
                                          criterion=self.criterion_map[individuo[2]], n_estimators=individuo[3])
        elif self.tipo_algoritmo == 3:
            return SVC(kernel=self.kernel_map[individuo[0]], C=individuo[1])
        elif self.tipo_algoritmo == 4:
            return GradientBoostingClassifier(n_estimators=individuo[0], learning_rate=individuo[1], max_depth=individuo[2])
        elif self.tipo_algoritmo == 5:
            return LogisticRegression(C=individuo[0], max_iter=500, solver="lbfgs")

    def fitness(self, individuo):
        modelo = self.gerar_modelo(individuo)
        modelo.fit(self.x_treinamento, self.y_treinamento)
        previsoes = modelo.predict(self.x_teste)
        return accuracy_score(self.y_teste, previsoes) * 100

    def selecao_por_torneio(self, populacao):
        pos1, pos2, pos3, pos4 = [randint(0, self.num_individuos - 1) for _ in range(4)]
        pai = populacao[pos1] if self.fitness(populacao[pos1]) > self.fitness(populacao[pos2]) else populacao[pos2]
        mae = populacao[pos3] if self.fitness(populacao[pos3]) > self.fitness(populacao[pos4]) else populacao[pos4]
        return list(pai), list(mae)

    def crossover(self, pai, mae):
        for i in range(randint(1, len(pai))):
            pai[i], mae[i] = mae[i], pai[i]
        return list(pai), list(mae)

    def mutacao(self, filho, filha):
        for i in range(len(filho)):
            if randint(1, 100) <= self.chance_de_mutar:
                filho[i] = self.gerar_individuo()[i]
            if randint(1, 100) <= self.chance_de_mutar:
                filha[i] = self.gerar_individuo()[i]
        return list(filho), list(filha)

    def executar(self):
        self.definir_xy()
        populacao = self.gerar_populacao()

        with open("GA.txt", "w") as file:
            file.write("1* Population:\n")
            for i in range(self.num_individuos // 2):
                pai, mae = self.selecao_por_torneio(populacao)
                filho, filha = self.crossover(pai, mae)
                filho_mutado, filha_mutada = self.mutacao(filho, filha)
                fitness1 = self.fitness(filho_mutado)
                fitness2 = self.fitness(filha_mutada)

                file.write(f"Individual: {filho_mutado} --> Accuracy {fitness1:.1f}%\n")
                file.write(f"Individual: {filha_mutada} --> Accuracy {fitness2:.1f}%\n")

                if fitness1 > self.melhor_acuracia:
                    self.melhor_acuracia = fitness1
                    self.melhor_individuo = list(filho_mutado)
                if fitness2 > self.melhor_acuracia:
                    self.melhor_acuracia = fitness2
                    self.melhor_individuo = list(filha_mutada)

            for j in range(1, self.num_populacoes):
                file.write(f"\n{j + 1}* Population:\n")
                for i in range(self.num_individuos // 2):
                    pai, mae = self.selecao_por_torneio(populacao)
                    filho, filha = self.crossover(pai, mae)
                    filho_mutado, filha_mutada = self.mutacao(filho, filha)
                    fitness1 = self.fitness(filho_mutado)
                    fitness2 = self.fitness(filha_mutada)

                    file.write(f"Individual: {filho_mutado} --> Accuracy {fitness1:.1f}%\n")
                    file.write(f"Individual: {filha_mutada} --> Accuracy {fitness2:.1f}%\n")

                    if fitness1 > self.melhor_acuracia:
                        self.melhor_acuracia = fitness1
                        self.melhor_individuo = list(filho_mutado)
                    if fitness2 > self.melhor_acuracia:
                        self.melhor_acuracia = fitness2
                        self.melhor_individuo = list(filha_mutada)

            file.write(f"\nBest Individual: {self.melhor_individuo} --> Best Accuracy: {self.melhor_acuracia:.1f}%")

# Exemplo de execução (ajuste o tipo_algoritmo conforme o desejado)
ga = GeneticAlgorithm(10, 10, 10, 3, "Dataset_com_temperatura_final.parquet")
