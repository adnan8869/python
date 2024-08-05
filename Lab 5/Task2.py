from collections import deque

def bfs_find_target(graph, start, target):
    explored = set()
    frontier = deque([start])
    
    while frontier:
        node = frontier.popleft()
        print(f"Visiting {node}")
        
        if node == target:
            print(f"Found target {target}")
            return True
        
        explored.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in explored:
                frontier.append(neighbor)
    
    print(f"Target {target} not found")
    return False

# Example graph represented as an adjacency list
graph = {
    'A': ['B', 'F', 'D', 'E'],
    'B': ['K', 'J'],
    'F': [],
    'D': ['G'],
    'E': ['C', 'H', 'I'],
    'K': ['N', 'M'],
    'J': [],
    'G': ['D'],
    'C': [],
    'H': [],
    'I': ['L'],
    'N': [],
    'M': [],
    'L': [],
}

# Starting BFS from vertex 'A' to find 'L'
print("BFS traversal to find 'L' starting from vertex 'A':")
bfs_find_target(graph, 'A', 'G')