from random import randint, uniform  # Importa as funções randint e uniform do módulo random
from pandas import read_csv  # Importa a função read_csv do módulo pandas
from sklearn.neighbors import KNeighborsClassifier  # Importa o classificador KNN do scikit-learn
from sklearn.tree import DecisionTreeClassifier  # Importa o classificador Decision Tree do scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Importa o classificador Random Forest do scikit-learn
from sklearn.model_selection import train_test_split  # Importa a função train_test_split do scikit-learn para dividir os dados em treino e teste
from sklearn.metrics import accuracy_score  # Importa a função accuracy_score do scikit-learn para calcular a acurácia
from sklearn.svm import SVC  # Importa o classificador SVM do scikit-learn

# Função para somar dois vetores, com uma condição específica para a segunda posição
def somar(parcela1, parcela2, tipo_algoritmo):
    soma = []
    for i in range(len(parcela1)):
        if tipo_algoritmo > 0 and i == 1:  # Se o tipo de algoritmo for maior que 0 e estiver na segunda posição
            soma.append(round(parcela1[i] + parcela2[i], 4))  # Arredonda a soma para 4 casas decimais
        else:
            soma.append(parcela1[i] + parcela2[i])
    return soma

# Função para subtrair dois vetores
def subtrair(minuendo, subtraendo):
    sub = []
    for i in range(len(minuendo)):
        sub.append(minuendo[i] - subtraendo[i])
    return sub

# Função para multiplicar um vetor por uma constante, com uma condição específica para a segunda posição
def multiplicar(vetor, c, tipo_algoritmo):
    mult = []
    for i in range(len(vetor)):
        if tipo_algoritmo > 0 and i == 1:  # Se o tipo de algoritmo for maior que 0 e estiver na segunda posição
            mult.append(vetor[i] * c,)  # Multiplica o valor pelo coeficiente e adiciona à lista
        else:
            mult.append(round(vetor[i] * c))  # Arredonda o resultado da multiplicação e adiciona à lista
    return mult

# Classe para representar uma partícula no algoritmo PSO
class Particula:
    def __init__(self, pos, vel):
        self.pos = pos  # Posição da partícula
        self.vel = vel  # Velocidade da partícula
        self.perf = None  # Desempenho da partícula
        self.c1 = uniform(0, 2)  # Coeficiente de aprendizado 1
        self.c2 = uniform(0, 2)  # Coeficiente de aprendizado 2
        self.melhor_pos = None  # Melhor posição encontrada pela partícula
        self.melhor_perf = None  # Melhor desempenho encontrado pela partícula

    def __str__(self):
        return 'Position: {} --> Speed: {} --> Accurracy: {:.1f}%'.format(self.pos, self.vel, self.perf)
    
    # Método para atualizar a velocidade da partícula
    def prox_vel(self, melhor_pos_geral, tipo_algoritmo):
        sub1 = subtrair(self.melhor_pos, self.pos)  # Vetor de diferença entre a posição atual e a melhor posição individual
        sub2 = subtrair(melhor_pos_geral, self.pos)  # Vetor de diferença entre a posição atual e a melhor posição global
        mult1 = multiplicar(sub1, self.c1, tipo_algoritmo)  # Multiplica o vetor de diferença 1 pelo coeficiente de aprendizado 1
        mult2 = multiplicar(sub2, self.c2, tipo_algoritmo)  # Multiplica o vetor de diferença 2 pelo coeficiente de aprendizado 2
        soma = somar(mult1, mult2, tipo_algoritmo)  # Soma os dois resultados
        self.vel = somar(soma, self.vel, tipo_algoritmo)  # Atualiza a velocidade da partícula

# Classe para implementar o algoritmo PSO
class PSO:
    def __init__(self, tam_enxame, num_interacoes, tipo_algoritmo, endereco_csv):
        self.tam_enxame = tam_enxame  # Tamanho do enxame (número de partículas)
        self.num_interacoes = num_interacoes  # Número de iterações do algoritmo
        self.tipo_algoritmo = tipo_algoritmo  # Tipo de algoritmo a ser utilizado
        self.dataframe = read_csv(endereco_csv)  # Dataframe carregado a partir do arquivo CSV
        self.melhor_pos_geral = None  # Melhor posição global encontrada pelo algoritmo
        self.melhor_perf_geral = 0  # Melhor desempenho global encontrado pelo algoritmo
        # Mapeamento dos parâmetros de acordo com o tipo de algoritmo
        self.weights_map = {0 : 'uniform', 1 : 'distance'}
        self.criterion_map = {0 : 'entropy', 1 : 'gini', 2 : 'log_loss'}
        self.kernel_map = {0 : 'sigmoid', 1 : 'poly', 2 : 'rbf', 3 : 'linear'}
        # Valores mínimos e máximos dos parâmetros de acordo com o tipo de algoritmo
        self.min = [[1, 0], [3, 0.1, 0], [3, 0.1, 0, 3], [0, 0.1]]
        self.max = [[20, 1], [20, 0.3, 2], [20, 0.3, 2, 200], [3, 1]]
        self.executar()  # Executa o algoritmo

    # Método para definir os conjuntos de dados de treinamento e teste
    def definir_xy(self):
        num_colunas = len(self.dataframe.columns)
        x = self.dataframe.iloc[:, 0 : num_colunas - 1].values
        y = self.dataframe.iloc[:, num_colunas - 1].values
        self.x_treinamento, self.x_teste, self.y_treinamento, self.y_teste = train_test_split(x, y, test_size = 0.25, random_state = 0)

    # Método para gerar uma posição inicial para as partículas
    def pos_zero(self):
        zero = []
        for i in range(len(self.max[self.tipo_algoritmo])):
            zero.append(self.max[self.tipo_algoritmo][i] // 2)
        return zero

    # Método para gerar uma posição aleatória para as partículas de acordo com o tipo de algoritmo
    def pos_aleatoria(self):
        if self.tipo_algoritmo == 0:
            return [randint(1, 20), randint(0, 1)]
        elif self.tipo_algoritmo == 1:
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2)]
        elif self.tipo_algoritmo == 2:
            return [randint(3, 20), round(uniform(0.1, 0.3), 4), randint(0, 2), randint(3, 200)]
        else:
            return[randint(0, 3), round(uniform(0.1, 1), 4)]
    
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
        if self.tipo_algoritmo == 0:
            return KNeighborsClassifier(n_neighbors = p.pos[0], weights = self.weights_map[p.pos[1]])
        elif self.tipo_algoritmo == 1:
            return DecisionTreeClassifier(max_depth = p.pos[0], min_samples_split = p.pos[1], criterion = self.criterion_map[p.pos[2]])
        elif self.tipo_algoritmo == 2:
            return RandomForestClassifier(max_depth = p.pos[0], min_samples_split = p.pos[1], criterion = self.criterion_map[p.pos[2]], n_estimators = p.pos[3])
        else:
            return SVC(kernel = self.kernel_map[p.pos[0]], C = p.pos[1])
        
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
                file.write(str(enxame[i]) + '\n')
            
            for j in range(1, self.num_interacoes):
                file.write("\n{}* Interation:\n".format(j + 1))
                
                for i in range(self.tam_enxame):
                    self.mover(enxame[i])
                    file.write(str(enxame[i]) + '\n')

            file.write('\nBest Position: {}\nBest Accuracy: {:.1f}%'.format(self.melhor_pos_geral, self.melhor_perf_geral))

# Criando uma instância da classe PSO com os parâmetros específicos
pso = PSO(10, 10, 3, 'heart.csv')