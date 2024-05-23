import numpy as np
from random import randint
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class Formiga:
    def __init__(self):
        # Inicialização dos atributos alpha e beta com valores aleatórios entre 1 e 3
        self.alpha = randint(1, 3)
        self.beta = randint(1, 3)
        self.acuracia = None  # Acurácia inicialmente definida como None
        self.caminho = []  # Lista para armazenar o caminho percorrido pela formiga
        self.indices = []  # Lista para armazenar os índices correspondentes ao caminho

    def __str__(self):
        # Método para representar a formiga como uma string
        return 'Caminho: {}, Alpha: {}, Beta: {} --> Acuracia: {:.1f}%'.format(self.caminho, self.alpha, self.beta, self.acuracia)

class AntColonyOptimization:
    def __init__(self, num_formigas, tipo_algoritmo, endereco_csv):
        # Método inicializador da classe AntColonyOptimization
        self.num_formigas = num_formigas  # Número de formigas na colônia
        self.tipo_algoritmo = tipo_algoritmo  # Tipo de algoritmo a ser usado
        self.dataframe = read_csv(endereco_csv)  # Carrega os dados do arquivo CSV em um DataFrame
        self.melhor_acuracia = 0  # Melhor acurácia encontrada até o momento
        self.melhor_formiga = None  # Melhor formiga encontrada até o momento
        # Mapeamento de parâmetros para os algoritmos de classificação
        self.weights_map = {0: 'uniform', 1: 'distance'}
        self.criterion_map = {0: 'entropy', 1: 'gini', 2: 'log_loss'}
        self.kernel_map = {0: 'sigmoid', 1: 'poly', 2: 'rbf', 3: 'linear'}
        # Definição de matrizes para determinar as arestas, valores mínimos e máximos para cada tipo de algoritmo
        self.matriz_arestas = [[20, 2], [18, 5, 3], [18, 5, 3, 198], [4, 19]]
        self.min = [[1, 0], [3, 0.1, 0], [3, 0.1, 0, 3], [0, 0.1]]
        self.max = [[20, 1], [20, 0.3, 2], [20, 0.3, 2, 200], [3, 1]]
        self.executar()  # Chama o método para executar o algoritmo

    def definirXY(self):
        # Método para definir os conjuntos X e Y a partir dos dados do DataFrame
        num_colunas = len(self.dataframe.columns)
        x = self.dataframe.iloc[:, 0: num_colunas - 1].values
        y = self.dataframe.iloc[:, num_colunas - 1].values
        # Divisão dos dados em conjuntos de treinamento e teste
        self.x_treinamento, self.x_teste, self.y_treinamento, self.y_teste = train_test_split(x, y, test_size=0.25, random_state=0)

    def definirDistancias(self):
        # Método para definir as distâncias entre os nós (arestas)
        self.qtd_conj_arestas = len(self.matriz_arestas[self.tipo_algoritmo])
        self.distancias = []

        for i in range(self.qtd_conj_arestas):
            self.distancias.append([])
            for j in range(self.matriz_arestas[self.tipo_algoritmo][i]):
                self.distancias[i].append(0.5)

    def definirFeromonios(self):
        # Método para definir os níveis iniciais de feromônios nas arestas
        self.feromonios = []

        for i in range(self.qtd_conj_arestas):
            self.feromonios.append([])
            for j in range(self.matriz_arestas[self.tipo_algoritmo][i]):
                self.feromonios[i].append(5)

    def gerarFormiga(self):
        # Método para gerar uma nova formiga com valores aleatórios de alpha e beta
        f = Formiga()
        return f
    
    def gerarFormigueiro(self):
        # Método para gerar um conjunto de formigas
        formigueiro = []
        for i in range(self.num_formigas):
            formigueiro.append(self.gerarFormiga())
        return formigueiro
    
    def formigasIniciais(self):
        # Método para gerar um conjunto inicial de formigas
        formigueiro = []
        for i in range(10):  # Número arbitrário de formigas iniciais
            formigueiro.append(self.gerarFormiga())
        return formigueiro
    
    def moverFormiga(self, f):
        # Método para mover uma formiga ao longo do caminho
        for i in range(self.qtd_conj_arestas):
            possibilidades = []
            somatorio = 0
 
            for j in range(self.matriz_arestas[self.tipo_algoritmo][i]):
                # Cálculo do somatório para as possibilidades de movimento
                somatorio += self.feromonios[i][j] ** f.alpha * self.distancias[i][j] ** f.beta
            
            for j in range(self.matriz_arestas[self.tipo_algoritmo][i]):
                # Cálculo das probabilidades de movimento
                possi = (self.feromonios[i][j] ** f.alpha * self.distancias[i][j] ** f.beta) / somatorio
                possibilidades.append(possi)

            # Escolha do próximo índice baseado nas probabilidades
            indice_sorteado = np.random.choice(self.matriz_arestas[self.tipo_algoritmo][i], p=possibilidades)

            if self.tipo_algoritmo > 0 and i == 1:
                # Ajuste dos valores de acordo com o tipo de algoritmo e aresta
                arestas_disponiveis = list(np.arange(self.min[self.tipo_algoritmo][i], self.max[self.tipo_algoritmo][i] + 0.05, 0.05))
                arestas_disponiveis = [round(num, 2) for num in arestas_disponiveis]
            else:
                arestas_disponiveis = list(range(self.min[self.tipo_algoritmo][i], self.max[self.tipo_algoritmo][i] + 1))
            
            f.caminho.append(arestas_disponiveis[indice_sorteado])
            f.indices.append(indice_sorteado)

    def gerarModelo(self, f):
        # Método para gerar um modelo de classificação com base no caminho da formiga
        if self.tipo_algoritmo == 0:
            return KNeighborsClassifier(n_neighbors=f.caminho[0], weights=self.weights_map[f.caminho[1]])
        elif self.tipo_algoritmo == 1:
            return DecisionTreeClassifier(max_depth=f.caminho[0], min_samples_split=f.caminho[1], criterion=self.criterion_map[f.caminho[2]])
        elif self.tipo_algoritmo == 2:
            return RandomForestClassifier(max_depth=f.caminho[0], min_samples_split=f.caminho[1], criterion=self.criterion_map[f.caminho[2]], n_estimators=f.caminho[3])
        else:
            return SVC(kernel=self.kernel_map[f.caminho[0]], C=f.caminho[1])
        
    def fitness(self, f):
        # Método para calcular a acurácia do modelo gerado pela formiga
        modelo = self.gerarModelo(f)
        modelo.fit(self.x_treinamento, self.y_treinamento)
        previsoes = modelo.predict(self.x_teste)
        return accuracy_score(self.y_teste, previsoes)
    
    def atualizarDistanciasFeromonios(self, f):
        # Método para atualizar as distâncias e os feromônios com base na acurácia do modelo
        acuracia = self.fitness(f)
        
        if acuracia > self.melhor_acuracia:
            self.melhor_acuracia = acuracia
            self.melhor_formiga = f

        for i in range(self.qtd_conj_arestas):
            # Atualização das distâncias e dos feromônios
            self.distancias[i][f.indices[i]] = (self.distancias[i][f.indices[i]] + acuracia) / 2
            self.feromonios[i][f.indices[i]] += 5
        
        f.acuracia = acuracia * 100
    
    def evaporacao(self):
        # Método para realizar a evaporacao dos feromônios
        for i in range(self.qtd_conj_arestas):
            for j in range(self.matriz_arestas[self.tipo_algoritmo][i]):
                self.feromonios[i][j] *= 0.9
                
                if self.feromonios[i][j] < 5:
                    self.feromonios[i][j] = 5

    def executar(self):
        # Método principal para executar o algoritmo
        self.definirXY()  # Define os conjuntos de treinamento e teste
        self.definirDistancias()  # Define as distâncias entre os nós
        self.definirFeromonios()  # Define os níveis iniciais de feromônios

        with open('ACO.txt', 'w') as file:
            formigueiro = self.formigasIniciais()  # Gera as formigas iniciais
            file.write('Movimento Inicial:\n')

            for i in range(10):  # Número arbitrário de formigas iniciais
                self.moverFormiga(formigueiro[i])
            
            for i in range(10):  # Número arbitrário de formigas iniciais
                # Atualiza as distâncias, feromônios e acurácias das formigas iniciais
                self.atualizarDistanciasFeromonios(formigueiro[i])
                file.write(str(formigueiro[i]) + '\n')
            # Fim do movimento inicial

            formigueiro = self.gerarFormigueiro()  # Gera um novo conjunto de formigas
            contador = 0
            file.write('\nMovimento Geral:\n')

            for i in range(self.num_formigas):  # Para cada formiga no formigueiro
                f = formigueiro[i]
                self.moverFormiga(f)  # Move a formiga ao longo do caminho
                self.atualizarDistanciasFeromonios(f)  # Atualiza as distâncias e feromônios
                self.evaporacao()  # Realiza a evaporacao dos feromônios
                contador += 1
                file.write(str(f) + '\n' + str(self.feromonios))
                
                if contador % 10 == 0:
                    file.write('\n')  # Adiciona uma linha em branco a cada 10 formigas
            # Fim do movimento geral

            file.write('\nMelhor Formiga:\n{}'.format(self.melhor_formiga))  # Escreve a melhor formiga encontrada

aco = AntColonyOptimization(200, 1, 'diabetes.csv')