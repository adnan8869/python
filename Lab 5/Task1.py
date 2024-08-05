import networkx as nx
import matplotlib.pyplot as plt

graph = {
    0: [1, 4],
    1: [0, 2, 3, 4],
    2: [1, 3],
    3: [1, 2, 4],
    4: [0, 3]
}

G = nx.Graph()

for node in graph:
    for neighbor in graph[node]:
        G.add_edge(node, neighbor)

nx.draw(G, with_labels=True)

plt.show()