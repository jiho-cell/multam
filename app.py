import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class AntSandpileNetwork:

    def __init__(
        self,
        num_nodes=6,
        capacity_limit=4,
        evaporation_rate=0.1,
        alpha=1.0,
        beta=1.0,
    ):
        """num_nodes: 사전 정의할 노드의 수 (고정)

        capacity_limit: 노드가 견딜 수 있는 임계 부하량 (모래산 모델)
        evaporation_rate: 페로몬 증발 비율 alpha, beta: 페로몬과 가용 용량의 가중치
        """
        self.num_nodes = num_nodes
        self.capacity_limit = capacity_limit
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta

        # 1. 고정된 노드 수 기반의 그래프 생성 (다이아몬드형/사다리형 네트워크 구성)
        self.G = nx.DiGraph()
        self._build_network()

    def _build_network(self):
        """노드 수와 연결 구조를 설정합니다."""
        # 노드 추가 (0부터 num_nodes - 1까지)
        for i in range(self.num_nodes):
            self.G.add_node(i, load=0)  # load: 현재 노드의 개미 수(부하)

        # 경로(Edge) 연결 및 초기 페로몬 설정 (계획서의 사다리형 노드 구조)
        # 예시: 0(입구) -> 1, 2 -> 3, 4 -> 5(출구) 구조
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 5), (4, 5)]

        for u, v in edges:
            if u < self.num_nodes and v < self.num_nodes:
                self.G.add_edge(u, v, pheromone=1.0)  # 초기 페로몬값 = 1.0

    def step(self, num_ants_entering=2):
        """1 스텝 진행: 개미 투입 -> 페로몬 선택 이동 -> 임계치 초과 시 연쇄 붕괴"""

        # --- Phase 1: 개미 이동 (Motter-Lai: 페로몬 기반 길 선택) ---
        current_node = 0
        for _ in range(num_ants_entering):
            curr = current_node
            while curr != self.num_nodes - 1:  # 출구 노드 도착 전까지
                neighbors = list(self.G.successors(curr))
                if not neighbors:
                    break

                # 이동 확률 계산 (페로몬 농도 비례)
                pheromones = np.array(
                    [self.G[curr][nbr]["pheromone"] for nbr in neighbors]
                )
                probabilities = (pheromones**self.alpha) / np.sum(
                    pheromones**self.alpha
                )

                # 확률에 따른 다음 노드 선택
                next_node = np.random.choice(neighbors, p=probabilities)

                # 선택된 경로에 페로몬 축적 (양의 피드백)
                self.G[curr][next_node]["pheromone"] += 0.5
                curr = next_node

                # 해당 노드에 개미 도착 (부하 증가)
                self.G.nodes[curr]["load"] += 1

        # --- Phase 2: 연쇄 붕괴 (Sandpile Toppling 메커니즘) ---
        cascade_occurred = False
        topple_queue = [
            n
            for n in self.G.nodes()
            if self.G.nodes[n]["load"] >= self.capacity_limit
        ]

        while topple_queue:
            node = topple_queue.pop(0)
            if self.G.nodes[node]["load"] >= self.capacity_limit:
                cascade_occurred = True
                # 부하 분산 (이웃 노드로 부하 넘김)
                neighbors = list(self.G.successors(node)) + list(
                    self.G.predecessors(node)
                )
                if neighbors:
                    excess = self.G.nodes[node]["load"]
                    self.G.nodes[node]["load"] = 0  # 노드 붕괴 후 초기화
                    share = excess // len(neighbors)

                    for nbr in neighbors:
                        self.G.nodes[nbr]["load"] += share
                        if (
                            self.G.nodes[nbr]["load"] >= self.capacity_limit
                            and nbr not in topple_queue
                        ):
                            topple_queue.append(nbr)

        # --- Phase 3: 페로몬 자연 증발 ---
        for u, v in self.G.edges():
            self.G[u][v]["pheromone"] *= 1 - self.evaporation_rate

        return cascade_occurred

    def draw_network(self, step_num):
        """네트워크 시각화"""
        pos = nx.spring_layout(self.G, seed=42)
        loads = [self.G.nodes[n]["load"] for n in self.G.nodes()]
        pheromones = [
            self.G[u][v]["pheromone"] * 2 for u, v in self.G.edges()
        ]

        plt.figure(figsize=(7, 5))
        nx.draw_networkx_nodes(
            self.G,
            pos,
            node_color=loads,
            cmap=plt.cm.Reds,
            node_size=700,
            vmin=0,
            vmax=self.capacity_limit + 2,
        )
        nx.draw_networkx_edges(
            self.G, pos, width=pheromones, edge_color="gray", arrows=True
        )
        nx.draw_networkx_labels(
            self.G,
            pos,
            labels={n: f"N{n}\n({self.G.nodes[n]['load']})" for n in self.G.nodes()},
        )

        plt.title(f"Step {step_num}: Red = Load, Edge Width = Pheromone")
        plt.axis("off")
        plt.show()


# --- 실행 부 ---
if __name__ == "__main__":
    # 노드 수를 6개로 지정하여 네트워크 생성
    sim = AntSandpileNetwork(num_nodes=6, capacity_limit=4)

    print("=== 시뮬레이션 시작 ===")
    for step in range(1, 6):
        is_cascade = sim.step(num_ants_entering=3)
        print(f"[Step {step}] 연쇄 붕괴(Cascade) 발생 여부: {is_cascade}")
        sim.draw_network(step)
