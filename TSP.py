import random
import numpy as np
# TSP https://developers.google.com/optimization/routing/tsp#python

# 0. New York - 1. Los Angeles - 2. Chicago - 3. Minneapolis - 4. Denver - 5. Dallas - 6. Seattle - 7. Boston - 8. San Francisco - 9. St. Louis - 10. Houston - 11. Phoenix - 12. Salt Lake City
# New York is the start and end location. Distance is measured in miles.

USA13 = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]


class Individuo:
    def __init__(self, genes):
        self.genes = genes  
        self.fitness = None

    def __repr__(self):
         if self.fitness is None:
            return f"{self.genes} | fit=None"
         else:
            return f"{self.genes} | fit={self.fitness:.2f}"


class AG:
    def __init__(self, pop_size=50, num_cidades=13):
        self.pop_size = pop_size
        self.num_cidades = num_cidades
        self.pop = []

        for _ in range(self.pop_size):
            genes = self.gerar_rota()
            ind = Individuo(genes)
            self.pop.append(ind)

    def gerar_rota(self):
        cidades = list(range(1, self.num_cidades))
        random.shuffle(cidades)
        return [0] + cidades + [0]

    def avaliar_pop(self):
        for ind in self.pop:
            distancia = Distancia_total(ind.genes)
            ind.fitness = distancia 
    
    def torneio(self, tamanho_torneio=3):
        candidatos = random.sample(self.pop, tamanho_torneio)
        vencedor = min(candidatos, key=lambda ind: ind.fitness)  
        return vencedor
    
    
    def crossover_OX(self, pai1, pai2, taxa_crossover=0.9):
     """Aplica o crossover OX e retorna um novo filho válido"""
     if random.random() > taxa_crossover:
        # Sem crossover → retorna uma cópia do pai
        return Individuo(pai1.genes[:])

     genes1 = pai1.genes[1:-1]
     genes2 = pai2.genes[1:-1]

     p1, p2 = sorted(random.sample(range(len(genes1)), 2))
     filho = [None] * len(genes1)
     filho[p1:p2+1] = genes1[p1:p2+1]

     pos = (p2 + 1) % len(genes1)
     for cidade in genes2:
        if cidade not in filho:
            filho[pos] = cidade
            pos = (pos + 1) % len(genes1)

     novo_filho = [0] + filho + [0]

     if not eh_valida_rota(novo_filho):
        novo_filho = [0] + random.sample(range(1, self.num_cidades), self.num_cidades - 1) + [0]

     return Individuo(novo_filho)
    
    def mutacao_swap(self, individuo, taxa_mutacao=0.05):
     """Aplica mutação por troca em um indivíduo"""
     if random.random() < taxa_mutacao:
        genes = individuo.genes[1:-1]  
        i, j = random.sample(range(len(genes)), 2)
        genes[i], genes[j] = genes[j], genes[i]  # trocando as cidades
        individuo.genes = [0] + genes + [0]

        if not eh_valida_rota(individuo.genes):
            individuo.genes = [0] + random.sample(range(1, self.num_cidades), self.num_cidades - 1) + [0]
    
    def selecionar_elite(self, quantidade=5):
     """Retorna os 5 melhores indivíduos da população atual"""
     ordenados = sorted(self.pop, key=lambda ind: ind.fitness)
     return ordenados[:quantidade]

def Distancia_total(rota):
    total = 0
    if eh_valida_rota(rota) == False: 
        return  total

    for posicao in range(len(rota) - 1):
        cidade_atual = rota[posicao]
        prox_cidade = rota[posicao + 1]
        total += USA13[cidade_atual][prox_cidade]  # soma a distância
    return total


def eh_valida_rota(rota):
    if len(rota) == 14:
        rota_sem_extremos = rota[1:-1]  # removendo a cidade 0 do início e fim
        if 0 in rota_sem_extremos:
         return False
        for i in range(len(rota_sem_extremos)):
            for j in range(i + 1, len(rota_sem_extremos)):
                if rota_sem_extremos[i] == rota_sem_extremos[j]:
                    return False 
        return True  
    else:
        return False  
    
def executar_aug():
    pop_size = 50
    geracoes = 400
    
    # --- inicializa o AG ---
    ag = AG(pop_size=pop_size, num_cidades=13)
    ag.avaliar_pop()

    print("Iniciando execução do AG com TSP...\n")

    for geracao in range(geracoes):
        nova_pop = []

        # --- elitismo ---
        elite = ag.selecionar_elite(5)
        nova_pop.extend(elite)

        # --- gera novos indivíduos ---
        while len(nova_pop) < ag.pop_size:
            pai1 = ag.torneio()
            pai2 = ag.torneio()
            filho = ag.crossover_OX(pai1, pai2, taxa_crossover=0.9)
            
            ag.mutacao_swap(filho, taxa_mutacao=0.5)
            filho.fitness = Distancia_total(filho.genes)
            nova_pop.append(filho)

        # --- atualiza população ---
        ag.pop = nova_pop[:ag.pop_size]
        ag.avaliar_pop()

        melhor = min(ag.pop, key=lambda ind: ind.fitness)
        print(f"Geração {geracao+1:03d} | Menor distância: {melhor.fitness:.2f}")

    # --- resultado final ---
    melhor_global = min(ag.pop, key=lambda ind: ind.fitness)
    print("\nMelhor solução encontrada:")
    print(melhor_global)
    return melhor_global


if __name__ == "__main__":

   num_execucoes = 30
   resultados = []

for i in range(num_execucoes):
        print(f"\n=== Execução {i + 1} de {num_execucoes} ===")
        melhor = executar_aug()  
        resultados.append(melhor.fitness)

   
media_fitness = np.mean(resultados)
desvio_fitness = np.std(resultados)

print("\n===== RESULTADOS FINAIS =====")
print(f"Média do fitness: {media_fitness:.2f}")
print(f"Desvio padrão do fitness: {desvio_fitness:.2f}")
np.savetxt("AUG_TSP",resultados, fmt='%d')
