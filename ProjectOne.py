import functools
import random
import queue
import time
import matplotlib.pyplot as plt

@functools.total_ordering
class State:
    row = 0
    column = 0
    moves = 0
    parent = None
    board = [[]]

    def __init__(self, board, parent, moves, currRow, currColumn):
        self.board = board
        self.parent = parent
        self.moves = moves
        self.row = currRow
        self.column = currColumn


    def __lt__(self, other):
        stateOne = self.moves
        stateTwo = other.moves
        goal = (len(self.board) - 1) * 2
        stateOne = stateOne + (goal - self.row + self.column)
        stateTwo = stateTwo + (goal - other.row + other.column)

        return stateOne < stateTwo


    def __eq__(self, other):
        stateOne = self.moves
        stateTwo = other.moves
        goal = (len(self.board) - 1) * 2
        stateOne = stateOne + (goal - self.row + self.column)
        stateTwo = stateTwo + (goal - other.row + other.column)

        return stateOne == stateTwo


@functools.total_ordering
class WrapperCompareAStar(State):

    def __lt__(self, other):
        stateOne = self.moves
        stateTwo = other.moves
        goal = (len(self.board) - 1) * 2
        stateOne = stateOne + (goal - self.row + self.col)
        stateTwo = stateTwo + (goal - other.row + other.col)

        return stateOne < stateTwo


    def __eq__(self, other):
        stateOne = self.moves
        stateTwo = other.moves
        goal = (len(self.board) - 1) * 2
        stateOne = stateOne + (goal - self.row + self.col)
        stateTwo = stateTwo + (goal - other.row + other.col)

        return stateOne == stateTwo


@functools.total_ordering
class WrapperCompareSPF(State):
    def __it__(self, other):
        stateOne = self.moves
        stateTwo = other.moves
        return stateOne < stateTwo

    def __eq__(self, other):
        stateOne = self.moves
        stateTwo = other.mvoes
        return stateOne == stateTwo



def transState(state, action):
    row = state.row
    column = state.column

    nextState = State(state.board, state, state.moves + 1, row, column)

    # Action Based On The Cardinal Directions
    if action == 0:  # Up
        nextState.row -= state.board[row][column]
    if action == 1:  # Down
        nextState.row += state.board[row][column]
    if action == 2:  # Left
        nextState.column -= state.board[row][column]
    if action == 3:  # Right
        nextState.column += state.board[row][column]

    return nextState


# Valid Action Method
def validAction(state, action):
    newRow = state.row
    newColumn = state.column

    if action == 0:
        newRow = newRow - state.board[newRow][newColumn]
    if action == 1:
        newRow = newRow + state.board[newRow][newColumn]
    if action == 2:
        newColumn = newColumn - state.board[newRow][newColumn]
    if action == 3:
        newColumn = newColumn + state.board[newRow][newColumn]
    if newRow < 0 or (newRow >= len(state.board)):
        return False
    if newColumn < 0 or (newColumn >= len(state.board)):
        return False

    return True


def BFS(state, eval):
    initializeEval(eval)

    queue = []
    queue.append(state)

    while queue:
        newState = queue.pop()

        row = newState.row
        col = newState.column

        if newState.moves < eval[row][col]:
            eval[row][col] = newState.moves
        else:
            continue

        for x in range(0, 4):
            if validAction(newState, x):
                temp = transState(newState, x)
                queue.append(temp)



def SPF(state, eval):
    initializeEval(eval)
    pQueue = queue.PriorityQueue()
    pQueue.put(state)
    while not pQueue.empty():
        newState = pQueue.get()

        row = newState.row
        col = newState.column

        if newState.moves < eval[row][col]:
            eval[row][col] = newState.moves
        else:
            continue

        for x in range(0, 4):
            if validAction(newState, x):
                temp = transState(newState, x)
                pQueue.put(temp)



def hillClimb(state, eval, num):
    max = len(state.board)

    BFS(state, eval)

    evaluation = getEval(eval)
    evalNums = []
    plt.title('Hill Climbing')
    plt.xlabel('Number Of Iterations')
    plt.ylabel('Evaluation Value')
    for x in range(num):
        row = random.randrange(0, max)
        col = random.randrange(0, max)

        if row == (max - 1) and col == (max - 1):
            x -= 1
            continue

        newVal = random.randrange(0, max - 1) + 1
        oldVal = state.board[row][col]

        if oldVal == newVal:
            x -= 1
            continue

        state.board[row][col] = newVal

        BFS(state, eval)

        newEval = getEval(eval)
        evalNums.append(newEval)
        print("New Evaluation: ", newEval, " Old Evaluation: ", evaluation)
        plt.plot(evalNums)

        if newEval > evaluation:
            evaluation = newEval
            printBoard(state.board)
        else:
            state.board[row][col] = oldVal

    plt.show()
    return state



def Population(num):
    population = []
    for x in range(10000):
        board = createTable(num)
    population.append(board)

    return population


# def CrossOver(currPop):
#     for x in currPop:
#         if()


def popMutation(currPop, eval, num):
    for state in currPop:
        max = len(state.board)

        BFS(state, eval)

        evaluation = getEval(eval)

        for x in range(num):
            row = random.randrange(0, max)
            col = random.randrange(0, max)

            if row == (max - 1) and col == (max - 1):
                x -= 1
                continue

            newVal = random.randrange(0, max - 1) + 1
            oldVal = state.board[row][col]

            if oldVal == newVal:
                x -= 1
                continue

            state.board[row][col] = newVal

            BFS(state, eval)

            newEval = getEval(eval)

            print("New Evaluation: ", newEval, " Old Evaluation: ", evaluation)

            if newEval >= evaluation:
                evaluation = newEval

            else:
                state.board[row][col] = oldVal

    return state

# AStar Algorithm
def AStar(state, eval):
    initializeEval(eval)
    pQueue = queue.PriorityQueue()
    pQueue.put(state)
    while not pQueue.empty():
        newState = pQueue.get()

        row = newState.row
        col = newState.column

        if newState.moves < eval[row][col]:
            eval[row][col] = newState.moves
        else:
            continue

        for x in range(0,4):
            if validAction(newState, x):
                temp = transState(newState, x)
                pQueue.put(temp)


# Initialize Evaluation Board
def initializeEval(eval):
    eval_Board_Size = len(eval)

    for i in range(eval_Board_Size):
        for j in range(eval_Board_Size):
            eval[i][j] = 999

    return


# Print Path Algorithm
# def printPath(state):
#     while state:
#         print(state.row, " ", state.column)
#         state = state.parent


# Print Board Algorithm
def printBoard(board):
    board_Size = len(board)

    for i in range(board_Size):
        for j in range(board_Size):
            if board[i][j] == 999:
                print("X  ", end = " ")
            else:
                print(board[i][j], " ", end = " ")

        print()


# Get Evaluation Function Value
def getEval(eval):
    solvable = False
    eval_Board_Size = len(eval)

    if eval[eval_Board_Size-1][eval_Board_Size-1] != 999:
        solvable = True
        return eval[eval_Board_Size-1][eval_Board_Size-1]

    count = 0
    for i in range(eval_Board_Size):
        for j in range(eval_Board_Size):
            if eval[i][j] == 999:
                count -= 1

    return count


def createTable(n):

    #gameBoard = [[0 for x in range(n)] for y in range(n)]
    gameBoard = [[0] * n for i in range(n)]

    for i in range(n):
        for j in range(n):
            gameBoard[i][j] = random.randrange(1, max((n - 1) - i, i, (n - 1) - j, j))
    return gameBoard


def createEvalTable(n):
    evalTable = [[0] * n for i in range(n)]
    return evalTable



startTime = time.time()
tempBoard = [
    [9,4,5,7,5,4,4,3,2,4,5],
    [1,3,7,4,8,6,7,5,3,2,4],
    [3,1,4,1,5,3,2,6,5,1,6],
    [9,8,7,4,1,3,5,2,4,6,1],
    [5,8,7,3,4,2,4,6,2,3,5],
    [4,6,6,4,1,4,3,3,6,3,3],
    [5,5,1,5,4,3,5,1,7,1,8],
    [3,4,5,2,5,2,5,4,5,5,3],
    [3,1,5,2,3,2,1,5,5,3,8],
    [6,8,8,4,7,7,8,4,4,3,4],
    [2,1,9,6,4,3,6,3,9,1,1]
]

tempEval = [
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0]
]

initialTempBoard = State (tempBoard, None, 0, 0, 0)


# num = int(input("Enter A N Value: "))
# board = createTable(num)
# eval = createEvalTable(num)
# printBoard(board)
# print()
# initial = State(board, None, 0, 0, 0)
# BFS(initial, eval)
# print("Evaluation Function Result: ", getEval(eval))
# printBoard(eval)

# n = len(board) - 1

#
# print("Evaluation Function Result: ", getEval(eval))
#
# hillClimb(initial, eval, 150)
# AStar(initial, eval)
# BFS(initialTempBoard, tempEval)
# SPF(initialTempBoard, tempEval)
AStar(initialTempBoard, tempEval)
printBoard(tempEval)
print("Evaluation Function Result: ", getEval(tempEval))

print("---%s Seconds ---" % (time.time() - startTime))
