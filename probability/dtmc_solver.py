"""
    We will represent DTMC with directed graphs
    We will find irreducibility in O((n + m)^2) by BFSing from each node
    Every DTMC passed in will be automatically positive recurrent if the chain is irreducible
    otherwise not (a computer can't represent a non finite graph in its entirety anyways)

    lastly we can solve for the stationary distribution using linear algebra
"""
import numpy as np
from collections import deque, defaultdict

class DTMC:

    def __init__(self, edges: list[tuple]) -> None:
        self.adj_list: defaultdict(list) = defaultdict(list) # type: ignore
        self.is_irreducible: bool = False
        self.pos_recurrent: bool = False
        self.stat_dist: dict[int] = {}

        for (a, b, p) in edges:
            self.adj_list[a].append((b, p))
        self._search()
        self._calc_stat_dist()
 
    def _search(self) -> None:
        self.is_irreducible = True
        for start in self.adj_list.keys():
            vis = set()
            q = deque([start])

            while q:
                cur = q.popleft()
                for (nxt, p) in self.adj_list[cur]:
                    if nxt not in vis:
                        vis.add(nxt)
                        q.append(nxt)

            self.is_irreducible &= (len(vis) == len(self.adj_list.keys()))
        
        self.pos_recurrent = self.is_irreducible

    def _calc_stat_dist(self) -> None:
        """
            Esentially just need to solve Ax = b
            where A is the set of equations
            - pi_1 - a_1 * pi_1 - b_1 * pi_2 ...
            - ...
            - pi_1 + pi_2 + ... 
            
            and b is [0, 0, 0, ... , 1]
        """
        node_to_idx = {node: idx for idx, node in enumerate(self.adj_list.keys())}
        n = len(self.adj_list.keys())
        A = np.zeros((n, n))
        b = np.zeros(n)
        b[-1] = 1
        for node, idx in node_to_idx.items():
            A[idx][idx] = 1
            for (nxt, p) in self.adj_list[node]:
                A[idx][node_to_idx[nxt]] -= p
        A[-1] = np.ones(n)

        # forfeit the last row because its overdetermined

        try:
            dist_vector = np.linalg.solve(A, b)
            self.stat_dist = {node: dist_vector[idx] for node, idx in node_to_idx.items()}
        except:
            self.stat_dist = None

def test_one():
    dtmc = DTMC([(0, 0, 0.5), (0, 1, 0.5), (1, 0, 0.5), (1, 1, 0.5)])
    assert dtmc.is_irreducible == True
    assert dtmc.pos_recurrent == True
    assert dtmc.stat_dist == {0: 0.5, 1: 0.5}


if __name__ == "__main__":
    dtmc = DTMC([(0, 0, 0.5), (0, 1, 0.5), (1, 0, 0.5), (1, 1, 0.5)])
    test_one()
    print("All tests passed!")
