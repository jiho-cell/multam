import numpy as np
import networkx as nx

# 1. 페로몬 업데이트 (개미가 지나갈 때 + 시간이 지날 때)
def update_pheromones(graph, evaporation_rate=0.05):
    for u, v in graph.edges():
        # 페로몬 증발
        graph[u][v]['pheromone'] *= (1 - evaporation_rate)
        
# 2. 페로몬 기반 개미의 다음 노드 확률적 선택
def select_next_node(graph, current_node):
    neighbors = list(graph.neighbors(current_node))
    pheromones = np.array([graph[current_node][nbr]['pheromone'] for nbr in neighbors])
    capacities = np.array([graph.nodes[nbr]['capacity'] - graph.nodes[nbr]['load'] for nbr in neighbors])
    
    # 만약 용량이 가득 차면(Sandpile 과부하) 이동 확률 급감 (저항 증가)
    capacities = np.maximum(capacities, 0.01) 
    
    # 확률 계산 (페로몬이 높고 용량 여유가 있는 곳으로)
    probabilities = (pheromones ** 1.2) * (capacities ** 1.0)
    probabilities /= probabilities.sum() # 정규화
    
    return np.random.choice(neighbors, p=probabilities)
