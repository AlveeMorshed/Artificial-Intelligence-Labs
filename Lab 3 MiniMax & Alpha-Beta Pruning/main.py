import numpy as np


def minimax(position, depth, maximizingPlayer, totalComparisons):
    if depth == 0:
        totalComparisons[0] += 1
        return funds[int(position)]
    if maximizingPlayer:
        maximum = -np.inf
        for j in range(0, branches):
            value = minimax(int(position) * branches + j, depth - 1, False, totalComparisons)
            maximum = max(value, maximum)
        return maximum
    else:
        minimum = np.inf
        for j in range(0, branches):
            value = minimax(int(position) * branches + j, depth - 1, True, totalComparisons)
            minimum = min(value, minimum)
        return minimum


def alphaBetaPruning(position, depth, maximizingPlayer, alpha, beta, pruneCount):
    if depth == 0:
        return funds[int(position)]
    if maximizingPlayer:
        maximum = -np.inf
        for j in range(0, branches):
            value = alphaBetaPruning(int(position) * branches + j, depth-1, False, alpha, beta, pruneCount)
            maximum = max(value, maximum)
            alpha = max(alpha, value)
            if alpha >= beta:
                pruneCount[0] += 1
                break
        return maximum
    else:
        minimum = np.inf
        for j in range(0, branches):
            value = alphaBetaPruning(int(position) * branches + j, depth-1, True, alpha, beta, pruneCount)
            minimum = min(value, minimum)
            beta = min(beta, value)
            if alpha >= beta:
                pruneCount[0] += 1
                break
        return minimum


file = open('input.txt', mode='r')
file = file.readlines()
for i in file:
    i.strip()
turns = file[0]
branches = int(file[1])
rangeLimit = file[2].split(' ')
totalDepth = 2 ** int(turns)
totalLeaves = int(branches) ** int(totalDepth)
funds = np.random.randint(rangeLimit[0], rangeLimit[1], totalLeaves)
totalComparisons = np.zeros(2)
pruneCount = np.zeros(1)
minimax(0, totalDepth, True, totalComparisons)
maxAmount = alphaBetaPruning(0, totalDepth, True, -np.inf, np.inf, pruneCount)
totalComparisons[1] = totalLeaves - pruneCount
print("Depth:", totalDepth, "\nBranch:", branches, "\nTerminal States (Leaf Nodes):", totalLeaves,
      "\nMaximum amount:", maxAmount,
      "\nComparisons:", int(totalComparisons[0]), "\nComparisons:", int(totalComparisons[1]))