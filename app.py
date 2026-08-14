"""import numpy as np
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
    
    return np.random.choice(neighbors, p=probabilities)"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def run_opa_simulation(num_nodes=20, steps=100, growth_rate=1.02, upgrade_rate=1.5, fail_prob=0.05):
    # 1. 초기 네트워크 생성 (Random Graph)
    G = nx.erdos_renyi_graph(n=num_nodes, p=0.2, seed=42)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n=num_nodes, p=0.2)

    # 노드 속성 초기화: 초기 부하(load) 및 최대 수용 용량(capacity)
    for node in G.nodes():
        G.nodes[node]['load'] = np.random.uniform(5, 10)
        # 수용 용량은 초기 부하의 1.5배로 설정
        G.nodes[node]['capacity'] = G.nodes[node]['load'] * 1.5

    blackout_sizes = []

    for step in range(steps):
        # --- [장기 동역학 1] 지속적 수요 증가 ---
        for node in G.nodes():
            G.nodes[node]['load'] *= growth_rate

        # --- [단기 동역학 1] 무작위 정전/고장 발생 원인 투입 ---
        failed_nodes = set()
        for node in G.nodes():
            if np.random.rand() < fail_prob:
                failed_nodes.add(node)

        # --- [단기 동역학 2] 연쇄 붕괴 (Cascading Failure) ---
        cascade_queue = list(failed_nodes)
        total_failed_in_step = set(failed_nodes)

        while cascade_queue:
            curr = cascade_queue.pop(0)
            neighbors = list(G.neighbors(curr))
            
            # 고장난 노드의 부하를 이웃 노드들에게 등분 재분배
            if neighbors:
                shared_load = G.nodes[curr]['load'] / len(neighbors)
                G.nodes[curr]['load'] = 0  # 초기화
                
                for nbr in neighbors:
                    if nbr not in total_failed_in_step:
                        G.nodes[nbr]['load'] += shared_load
                        # 임계 용량을 초과하면 연쇄 붕괴 발생
                        if G.nodes[nbr]['load'] > G.nodes[nbr]['capacity']:
                            total_failed_in_step.add(nbr)
                            cascade_queue.append(nbr)

        # 기록: 이번 단계의 정전(붕괴) 규모
        blackout_sizes.append(len(total_failed_in_step))

        # --- [장기 동역학 2] 복구 및 용량 증설 (Upgrade) ---
        for node in G.nodes():
            if node in total_failed_in_step:
                # 정전을 겪은 노드의 용량을 확충
                G.nodes[node]['capacity'] *= upgrade_rate
                # 부하 재설정
                G.nodes[node]['load'] = np.random.uniform(5, 10)

    return blackout_sizes

# 시뮬레이션 실행
blackout_history = run_opa_simulation(num_nodes=30, steps=200)

# 시각화: 시간에 따른 정전 규모 및 붕괴 패턴
plt.figure(figsize=(10, 4))
plt.plot(blackout_history, color='crimson', marker='o', markersize=3, linestyle='-')
plt.title('OPA Model: Blackout Cascade Simulation')
plt.xlabel('Time Step (Long-term Scale)')
plt.ylabel('Number of Failed Nodes (Blackout Size)')
plt.grid(True)
plt.show()
