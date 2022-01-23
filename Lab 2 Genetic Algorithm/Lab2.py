import numpy as np
import random as rnd

def fitness(board):
    '''calculates the fitness score of each
       of the individuals in the population

      returns a 1D numpy array: index referring to
      ith individual in population, and value referring
      to the fitness score.'''
    fitness = 0
    attPair = 0
    for x in range(0, len(board)-1):
        for j in range(x+1, n ):
            if(int(board[x]) == int(board[j]) 
                or np.absolute(int(board[x])-int(board[j]) ) == np.absolute(x-j)):
                attPair = attPair + 1
   
    fitness = 28 - attPair
    return fitness


def select(population, fitPercent):
    ''' take input:  population and fit
      fit contains fitness values of each of the individuals 
      in the population  
      
      return:  one individual randomly giving
      more weight to ones which have high fitness score'''
    totalFitness = 0
    for  likelihood in fitPercent:
        totalFitness = totalFitness + likelihood
    rndThreshold = rnd.uniform(0, totalFitness)
    base = 0
    for board, likelihood in zip(population, fitPercent):
        if base + likelihood >= rndThreshold:
            return board
        base = base + likelihood


def crossover(x, y):
    '''take input: 2 parents - x, y.
       Generate a random crossover point.
       Append first half of x with second
       half of y to create the child

       returns: a child chromosome'''
    rndIdx = rnd.randint(0, 7)
    child = x[0:rndIdx] + y[rndIdx:len(x)]
    return child


def mutate(child):
    '''take input: a child
       mutates a random
       gene into another random gene

       returns: mutated child'''
    rndIdx = rnd.randint(0,7)
    child[rndIdx] = rnd.randint(0,7)
    return child

def probability(board, population):
    denominator = 0
    for perBoard in population:
        denominator = denominator + fitness(perBoard)
    fitPercent = fitness(board)/denominator
    return fitPercent

def GA(population, n, mutation_threshold):
    '''implement the pseudocode here by
       calling the necessary functions- Fitness,
       Selection, Crossover and Mutation

       print: the max fitness value and the
       chromosome that generated it which is ultimately
       the solution board'''
    locPopulation = []
    fitPercent = []
    for board in population:
        fitPercent.append(probability(board, population))
    for i in range(0, len(population)):
        x = select(population, fitPercent)
        y = select(population, fitPercent)
        child = crossover( x, y)
        if rnd.random() < mutation_threshold:
            child = mutate(child)
        locPopulation.append(child)
        if fitness(child) == 28:
            break
           

    return locPopulation

'''for 8 queen problem, n = 8'''
n = 8

'''start_population denotes how many individuals/chromosomes are there
  in the initial population n = 8'''
start_population = 10

'''if you want you can set mutation_threshold to a higher value,
   to increase the chances of mutation'''
mutation_threshold = 0.3

'''creating the population with random integers between 0 to 7 inclusive
   for n = 8 queen problem'''
population = [[rnd.randint(0, n) for qPos in range(n)] for unit in range(start_population)]
initial_pop = population
generation = 0
print("Running... please wait, the solution might be far away in future")
while not 28 in [fitness(board) for board in population] and generation != 100000:
    '''calling the GA function'''
    population = GA(population, n, mutation_threshold)
    generation = generation + 1
    fitMatrix = []
    for board2 in population:
        fitMatrix.append(fitness(board2))
    print("Generation: ",generation)
    print("Fitness Matrix: ", fitMatrix)
for board in population :
    if fitness(board) == 28:
        print("Strongest child ",board,"\nwith fitness ",fitness(board)," found after ",generation," generations.")

