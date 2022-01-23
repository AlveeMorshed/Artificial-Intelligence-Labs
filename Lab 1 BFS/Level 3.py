import numpy as np
from queue import Queue
from queue import LifoQueue

def bfs(gr, source, goal):
    visited = np.zeros(len(gr), dtype=bool)
    qu = Queue(maxsize=len(gr))
    qu.put(int(source))
    levelSize = qu.qsize()
    depth = 0
    for j in range(0, len(gr)):
        currentNode = qu.get()
        visited[currentNode] = True
        levelSize = levelSize - 1
        for k in range(0, len(goal)):
            if currentNode == goal[k] and depth < killTimes[k]:
                killTimes[k] = depth
                break

        for child in range(0, len(gr[currentNode])):
            if not visited[gr[currentNode][child]]:
                qu.put(gr[currentNode][child])
        if levelSize == 0:
            depth = depth + 1
            levelSize = qu.qsize()
        if qu.empty():
            break
    return depth

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



file = open('input3.txt', mode='r')
fileLine = file.readlines()
for i in file:
    i.strip()
vertices = int(fileLine[0])
edges = int(fileLine[1])
graph = {}
invertedGraph = {}
for i in range(2, edges+2):
    connection = fileLine[i].split(' ')
    vertex1, vertex2 = connection
    vertex1 = int(vertex1) ; vertex2 = int(vertex2)
    if vertex1 not in graph.keys():
        graph[vertex1] = list()
        invertedGraph[vertex1] = list()
    if vertex2 not in graph.keys():
        graph[vertex2] = list()
        invertedGraph[vertex2] = list()
    graph[vertex1].append(vertex2)
    invertedGraph[vertex2].append(vertex1)

linaLocation = fileLine[-7]
kPositions = np.empty(int(fileLine[-6]), dtype=int)
killTimes = np.empty(len(kPositions))
killTimes[:] = np.inf
for i in range(0, len(kPositions)):
    kPositions[i] = int(fileLine[-len(kPositions)+i])
bfs(invertedGraph, linaLocation, kPositions)
print(int(killTimes.min()))

dfs(invertedGraph, linaLocation)