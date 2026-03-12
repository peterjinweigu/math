'''
Specification:
simple solver for zero sum payoff matrix, returns one optimal strategy for P1
as well as the value of the game for P1
process ->
1. reduce the matrix
   uses basic techniques to first reduce the matrix
   actually there is some good reasoning here why algorithmically
   it makes sense to reduce first
2. identify saddle points, if any return here
3. solve the reduced matrix with equalizing payoffs
   this produces a system of linear equations
   which is rather suited for numpy to solve
if values are invalid for probabilities, assume the game is not solvable and return None
although there still may exist a solution at this point in time, in which
our solver is just not sophicated enough to find it

aside:
our domination techniques will be simpler than what is fully possible
just only check for single row/col dominations
do this until there are none left
full domination requires linear programming which may or may not be possible

lastly, something we learned in class is for 2xn matrices, you can technically
find a solution by analysing piecewise functions as a function of a single variable
which would increase the solution range of this algorithm
however maybe i leave this as an extra step for the future

future checklist (easiest to hardest):
- check for diagonal matrix
- check for anti symmetry -> get strategy??
- piecewise analysis on 2xn
'''

import numpy as np
from dataclasses import dataclass

@dataclass
class Solution:
    value: int
    strat: list[int]

    def __str__(self):
        return f"Value of the game is: {self.value}\nThe strategy is: {str(self.strat)}"

@dataclass
class GameMatrix:
    A: list[list[int]]
    rows: list[int]
    cols: list[int]


def a_geq_b(a: list[int], b: list[int]) -> bool:
    for (a_i, b_i) in zip(a, b):
        if a_i < b_i:
            return False
    return True


def reduce_matrix(payoff_matrix: GameMatrix):
    A = payoff_matrix.A
    rows = len(A)
    cols = len(A[0])
    A_T = list(zip(*A))

    for i in range(rows):
        for j in range(rows):
            if i == j:
                continue
            if a_geq_b(A[i], A[j]):
                payoff_matrix.A.pop(j)
                payoff_matrix.rows.pop(j)
                reduce_matrix(payoff_matrix)
                return
    
    for i in range(cols):
        for j in range(cols):
            if i == j:
                continue
            if a_geq_b(A_T[j], A_T[i]):
                for row in payoff_matrix.A:
                    row.pop(j)
                payoff_matrix.cols.pop(j)
                reduce_matrix(payoff_matrix)
                return


def find_sd(payoff_matrix: GameMatrix) -> tuple | None:
    A = payoff_matrix.A
    row_idx = payoff_matrix.rows
    col_idx = payoff_matrix.cols
    rows = len(A)
    cols = len(A[0])
    A_T = list(zip(*A))
    for row in range(rows):
        for col in range(cols):
            if A[row][col] == max(A_T[col]) and A[row][col] == min(A[row]):
                return (row_idx[row], col_idx[col])
    return None


def solve_system(payoff_matrix: GameMatrix) -> list[int] | None:
    '''
        Here we want to setup and solve the matrix B s.t.
        Solving Bx = y solves the system of equations
        x_1 + x_2 ... x_n = 1 (1 equation) this is the strategy of P1
        x^TA = v (n equations) here we assume A is a square matrix
        or x^TA - v = 0
        Thus x is the vector [x_1, x_2, ..., x_n, v] (n+1 length)
        And B is the matrix
        [
            1        1        ...  0 
            A_{1, 1} A_{1, 2} ... -1
            ...
            A_{n, 1} A_{n, 2} ... -1
        ]

        and y is the vector [1, 0, ... 0] (n+1 length)
    '''
    A = payoff_matrix.A
    # cannot solve non square matrix (without LP)
    if len(A) != len(A[0]):
        return None

    n = len(A)
    B = [[1 for i in range(n)]]
    B[-1].append(0)

    for row in A:
        B.append(row)
        B[-1].append(-1)
    
    try:
        return np.linalg.solve(B, [1] + [0 for i in range(n)]).tolist()
    except:
        return None
    

def solve_matrix(A: list[list[int]]) -> Solution | None:

    rows = len(A)
    cols = len(A[0])

    payoff_matrix = GameMatrix (
        A,
        [i for i in range(rows)],
        [i for i in range(cols)]
    )

    reduce_matrix(payoff_matrix)

    sd = find_sd(payoff_matrix)

    if sd != None:
        strat = [0 for row in range(rows)]
        strat[sd[0]] = 1
        return Solution(A[sd[0]][sd[1]], strat)

    out = solve_system(payoff_matrix)

    if out != None:
        value = out[-1]
        mini_strat = out[0:-1]
        strat = [0 for row in range(rows)]
        for (i, idx) in enumerate(payoff_matrix.rows):
            strat[idx] = mini_strat[i]
        return Solution(value, strat)

    return None

'''
Testing

Case 1 (Saddle point exists):
[
[1, 0, 4],
[2, 3, 3],
[1, 4, 0]
]

strat here is [0, 1, 0] with payoff 2

Case 2 (Equalize payoffs):
[
[1, 0],
[0, 2]
]

this is the hider and chooser game, for player 1
an optimal mixed strategy is [2/3, 1/3] with payout 2/3

Case 3 (Dominate then equalize payoffs):
[
[0, 0, -1, -2],
[2, 2, 0, 3],
[0, 1, 1, 3],
[-3, 3, 0, 3]
]

example from mt, domination reduces to 2x2 which is easily solved by equalizing payoffs
strat is [0, 1/3, 2/3, 0] with payoff 2/3

Case 4 (Equalize payoffs harder)
[
[0, 1, -2],
[-1, 0, 3],
[2, -3, 0]
]

strat is [3/6, 2/6, 1/6] with value 0
(actually this is an antisymmetric matrix)
'''

def test_one():
    matrix = [
        [1, 0, 4],
        [2, 3, 3],
        [1, 4, 0]
    ]
    print(solve_matrix(matrix))

def test_two():
    matrix = [
        [1, 0],
        [0, 2]
    ]
    print(solve_matrix(matrix))

def test_three():
    matrix = [
        [0, 0, -1, -2],
        [2, 2, 0, 3],
        [0, 1, 1, 3],
        [-3, 3, 0, 3]
    ]
    print(solve_matrix(matrix))

def test_four():
    matrix = [
        [0, 1, -2],
        [-1, 0, 3],
        [2, -3, 0]
    ]
    print(solve_matrix(matrix))

if __name__ == "__main__":
    test_one()
    test_two()
    test_three()
    test_four()