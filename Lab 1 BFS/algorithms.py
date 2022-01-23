import numpy as np
from queue import Queue
from queue import LifoQueue


def bfs(graph, source, goal):
    currentNode = int(source)
    visited = np.zeros(len(graph), dtype=bool)
    qu = Queue(maxsize=len(graph))
    qu.put(currentNode)
    levelSize = qu.qsize()
    depth = 0
    for i in range(0, len(graph)):
        currentNode = qu.get()
        visited[currentNode] = True
        levelSize = levelSize - 1
        if currentNode == goal:
            break
        else:
            for child in range(0, len(graph[currentNode])):
                if not visited[graph[currentNode][child]]:
                    qu.put(graph[currentNode][child])
            if levelSize == 0:
                depth = depth + 1
                levelSize = qu.qsize()
            if qu.empty():
                break
    return depth

# Still incomplete
def dfs(graph, source):
    visited = set()
    qu = LifoQueue(maxsize=len(graph))
    qu.put(source)
    while not qu.empty():
        currentNode = qu.get()
        if currentNode not in visited:
            print(currentNode)
            visited.add(currentNode)
            print(visited)
            qu.put(currentNode)
        for i in range(0, len(graph[currentNode])):
            if graph[currentNode][i] not in visited:
                qu.put(graph[currentNode][i])
                break

        else:
            qu.get()



