import heapq

# Heuristic: Misplaced tiles
def misplaced_tiles(state, goal):
    return sum(state[i][j] != goal[i][j] and state[i][j] != 0 for i in range(3) for j in range(3))

# Find the 0 position (empty space)
def find_zero(state):
    for i, row in enumerate(state):
        if 0 in row:
            return i, row.index(0)

# Generate neighbors by swapping the empty space
def get_neighbors(state):
    neighbors = []
    x, y = find_zero(state)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(new_state)
    return neighbors

# A* search
def a_star_search(start, goal):
    pq = [(misplaced_tiles(start, goal), 0, start, [])]
    visited = set()
    
    while pq:
        _, cost, current, path = heapq.heappop(pq)
        if current == goal:
            return path + [current]
        state_tuple = tuple(map(tuple, current))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(current):
            heapq.heappush(pq, (cost + 1 + misplaced_tiles(neighbor, goal), cost + 1, neighbor, path + [current]))

# Initial and goal states
start = [[1, 2, 3], [4, 0, 5], [7, 8, 6]]
goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# Run the search and print the result
solution = a_star_search(start, goal)
for step in solution:
    for row in step:
        print(row)
    print()
