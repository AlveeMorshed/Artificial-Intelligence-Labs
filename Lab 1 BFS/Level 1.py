import numpy as np
from queue import Queue


def bfs(graph, source, goal):
    visited = np.zeros(len(graph), dtype=bool)
    qu = Queue(maxsize=len(graph))
    qu.put(int(source))
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


file = open('input1.txt', mode='r')
file = file.readlines()
for i in file:
    i.strip()
vertices = int(file[0])
edges = int(file[1])
graph = {}
for i in range(2, edges+2):
    connection = file[i].split(' ')
    vertex1, vertex2 = connection
    vertex1 = int(vertex1) ; vertex2 = int(vertex2)
    if vertex1 not in graph.keys():
        graph[vertex1] = list()
    if vertex2 not in graph.keys():
        graph[vertex2] = list()
    graph[vertex1].append(vertex2)
    graph[vertex2].append(vertex1)
linaLocation = int(file[-1])
print(bfs(graph, 0, linaLocation))