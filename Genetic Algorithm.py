from random import randint, uniform  # Importa as funções randint e uniform do módulo random
from pandas import read_csv  # Importa a função read_csv do módulo pandas
from sklearn.neighbors import KNeighborsClassifier  # Importa o classificador KNN do scikit-learn
from sklearn.tree import DecisionTreeClassifier  # Importa o classificador Decision Tree do scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Importa o classificador Random Forest do scikit-learn
from sklearn.model_selection import train_test_split  # Importa a função train_test_split do scikit-learn para dividir os dados em treino e teste
from sklearn.metrics import accuracy_score  # Importa a função accuracy_score do scikit-learn para calcular a acurácia
from sklearn.svm import SVC  # Importa o classificador SVM do scikit-learn

class GeneticAlgorithm:
    def __init__(self, num_individuos, num_populacoes, chance_de_mutar, tipo_algortimo, endereco_csv):
        self.num_individuos = num_individuos  # Número de indivíduos na população
        self.num_populacoes = num_populacoes  # Número de gerações
        self.chance_de_mutar = chance_de_mutar  # Chance de mutação
        self.tipo_algoritmo = tipo_algortimo  # Tipo de algoritmo (0: KNN, 1: Decision Tree, 2: Random Forest, 3: SVM)
        self.dataframe = read_csv(endereco_csv)  # DataFrame carregado a partir do arquivo CSV
        self.melhor_individuo = None  # Melhor indivíduo encontrado durante a execução do algoritmo
        self.melhor_acuracia = 0  # Melhor acurácia encontrada durante a execução do algoritmo
        self.weights_map = {0 : 'uniform', 1 : 'distance'}  # Mapeamento de pesos para o KNN
        self.criterion_map = {0 : 'entropy', 1 : 'gini', 2 : 'log_loss'}  # Mapeamento de critérios para árvores de decisão
        self.kernel_map = {0 : 'sigmoid', 1 : 'poly', 2 : 'rbf', 3 : 'linear'}  # Mapeamento de kernels para SVM
        self.tam_modelos = [2, 3, 4]  # Número de parâmetros para cada tipo de modelo
        self.rand_modelos = [[randint(1, 20), randint(0, 1)], [randint(3, 20), randint(0, 2), uniform(0.1, 0.3)], 
                             [randint(3, 20), randint(0, 2), uniform(0.1, 0.3), randint(40, 100)], [randint(0, 4), uniform(0.1, 1)]]  # Parâmetros aleatórios iniciais para cada tipo de modelo
        self.executar()  # Chama o método para iniciar a execução do algoritmo

    # Método para dividir os dados em treino e teste
    def definir_xy(self):
        num_colunas = len(self.dataframe.columns)
        x = self.dataframe.iloc[:, 0 : num_colunas - 1].values
        y = self.dataframe.iloc[:, num_colunas - 1].values
        self.x_treinamento, self.x_teste, self.y_treinamento, self.y_teste = train_test_split(x, y, test_size = 0.25, random_state = 0)

    # Método para gerar um indivíduo com parâmetros aleatórios
    def gerar_individuo(self):
        if self.tipo_algoritmo == 0:
            return [randint(1, 20), randint(0, 1)]  # Parâmetros para KNN
        elif self.tipo_algoritmo == 1:
            return [randint(3, 20), randint(0, 2), round(uniform(0.1, 0.3), 4)]  # Parâmetros para Decision Tree
        elif self.tipo_algoritmo == 2:
            return [randint(3, 20), randint(0, 2), round(uniform(0.1, 0.3), 4), randint(40, 100)]  # Parâmetros para Random Forest
        else:
            return[randint(0, 3), round(uniform(0.1, 1), 4)]  # Parâmetros para SVM
      
    # Método para gerar uma população inicial
    def gerar_populacao(self):
        populacao = []
        for i in range(self.num_individuos):
            populacao.append(self.gerar_individuo())
        return populacao
    
    # Método para gerar um modelo a partir dos parâmetros de um indivíduo
    def gerar_modelo(self, individuo):
        if self.tipo_algoritmo == 0:
            return KNeighborsClassifier(n_neighbors = individuo[0], weights = self.weights_map[individuo[1]])  # KNN
        elif self.tipo_algoritmo == 1:
            return DecisionTreeClassifier(max_depth = individuo[0], criterion = self.criterion_map[individuo[1]], min_samples_split = individuo[2])  # Decision Tree
        elif self.tipo_algoritmo == 2:
            return RandomForestClassifier(max_depth = individuo[0], criterion = self.criterion_map[individuo[1]], min_samples_split = individuo[2], n_estimators = individuo[3])  # Random Forest
        else:
            return SVC(kernel = self.kernel_map[individuo[0]], C = individuo[1])  # SVM
    
    # Método para calcular a aptidão (acurácia) de um indivíduo
    def fitness(self, individuo):
        modelo = self.gerar_modelo(individuo)
        modelo.fit(self.x_treinamento, self.y_treinamento)
        previsoes = modelo.predict(self.x_teste)
        return accuracy_score(self.y_teste, previsoes) * 100
    
    # Método para seleção de pais por torneio
    def selecao_por_torneio(self, populacao):
        pos1 = randint(0, self.num_individuos - 1)
        pos2 = randint(0, self.num_individuos - 1)
        pos3 = randint(0, self.num_individuos - 1)
        pos4 = randint(0, self.num_individuos - 1)
        if self.fitness(populacao[pos1]) < self.fitness(populacao[pos2]):
            pos1 = pos2
        if self.fitness(populacao[pos3]) < self.fitness(populacao[pos4]):
            pos3 = pos4
        pai = list(populacao[pos1])
        mae = list(populacao[pos3])
        return pai, mae
    
    # Método de crossover
    def crossover(self, pai, mae):
        for i in range(randint(1, len(pai))):
            aux = pai[i]
            pai[i] = mae[i]
            mae[i] = aux
        filho = list(pai)
        filha = list(mae)
        return filho, filha
    
    # Método de mutação
    def mutacao(self, filho, filha):
        for i in range(len(filho)):
            chance1 = randint(1, 100)
            chance2 = randint(1, 100)
            if chance1 <= self.chance_de_mutar:
                filho[i] = round(self.rand_modelos[self.tipo_algoritmo][i], 4)
            if chance2 <= self.chance_de_mutar:
                filha[i] = round(self.rand_modelos[self.tipo_algoritmo][i], 4)
        filho_mutado = list(filho)
        filha_mutada = list(filha)
        return filho_mutado, filha_mutada
    
    # Método principal para executar o algoritmo genético
    def executar(self):
        self.definir_xy()  # Divide os dados em treino e teste
        populacao = self.gerar_populacao()  # Gera a população inicial

        with open("GA.txt", "w") as file:
            
            file.write('1* Population: \n')
            for i in range(self.num_individuos // 2):
                pai, mae = self.selecao_por_torneio(populacao)
                filho, filha = self.crossover(pai, mae)
                filho_mutado, filha_mutada = self.mutacao(filho, filha)
                fitness1 = self.fitness(filho_mutado)
                fitness2 = self.fitness(filha_mutada)
                file.write('Individual: {} --> Accuracy {:.1f}%\nIndividual: {} --> Accuracy {:.1f}%\n'.format(filho_mutado, fitness1, filha_mutada, fitness2))
                           
                if self.melhor_acuracia < fitness1:
                    self.melhor_acuracia = fitness1
                    self.melhor_individuo = list(filho_mutado)

                if self.melhor_acuracia < fitness2:
                    self.melhor_acuracia = fitness2
                    self.melhor_individuo = list(filha_mutada)
            
            for j in range(1, self.num_populacoes):
                file.write('\n{}* Population:\n'.format(j + 1))

                for i in range(self.num_individuos // 2):
                    pai, mae = self.selecao_por_torneio(populacao)
                    filho, filha = self.crossover(pai, mae)
                    filho_mutado, filha_mutada = self.mutacao(filho, filha)
                    fitness1 = self.fitness(filho_mutado)
                    fitness2 = self.fitness(filha_mutada)
                    file.write('Individual: {} --> Accuracy {:.1f}%\nIndividual: {} --> Accuracy {:.1f}%\n'.format(filho_mutado, fitness1, filha_mutada, fitness2))
                           
                    if self.melhor_acuracia < fitness1:
                        self.melhor_acuracia = fitness1
                        self.melhor_individuo = list(filho_mutado)

                    if self.melhor_acuracia < fitness2:
                        self.melhor_acuracia = fitness2
                        self.melhor_individuo = list(filha_mutada)

            file.write('\nBest Individual: {} --> Best Accuracy: {:.1f}%'.format(self.melhor_individuo, self.melhor_acuracia))

# Criando uma instância da classe GeneticAlgorithm com os parâmetros específicos
ga = GeneticAlgorithm(10, 10, 10, 3, 'diabetes.csv')