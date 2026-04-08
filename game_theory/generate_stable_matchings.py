from dataclasses import dataclass
from itertools import permutations

@dataclass
class Matching:
    matching: list[tuple]

    def latex(self):
        return f"$${','.join([f'({x}, {y})' for x, y in self.matching])}$$"

    def __str__(self):
        return ','.join([f"({x}, {y})" for x, y in self.matching])

    def __repr__(self):
        return ','.join([f"({x}, {y})" for x, y in self.matching])

def is_stable(l1: list[str], l2: list[str], rank1: dict[str, list], rank2: dict[str, list]) -> bool:
    n = len(l1)
    for i in range(n):
        for j in range(i+1, n):
            a, b, c, d = l1[i], l2[i], l1[j], l2[j]
            if rank1[a].index(d) < rank1[a].index(b) and rank2[d].index(a) < rank2[d].index(c):
                return False
            elif rank1[c].index(b) < rank1[c].index(d) and rank2[b].index(c) < rank2[b].index(a):
                return False
    return True


"""
    Function to generate all stable matchings in O(n!n^2) time
    where n is the number of candidates
    iterate over all pairings and check if stable
"""
def generate_matchings(a: dict[str, list], b: dict[str, list]) -> list[Matching]:
    matchings: list[Matching] = []
    l1 = list(a.keys())
    for l2 in permutations(list(b.keys())):
        if is_stable(l1, l2, a, b):
            matchings.append(Matching([(x, y) for (x, y) in zip(l1, l2)]))
    return matchings


def test():
    rank1 = {
        'a': ['p', 'q', 'r', 's'],
        'b': ['q', 's', 'r', 'p'],
        'c': ['p', 'r', 's', 'q'],
        'd': ['q', 's', 'p', 'r']
    }

    rank2 = {
        'p': ['b', 'c', 'a', 'd'],
        'q': ['a', 'd', 'b', 'c'],
        'r': ['a', 'c', 'd', 'b'],
        's': ['a', 'c', 'd', 'b']
    }

    matchings = generate_matchings(rank1, rank2)
    for m in matchings:
        print(m)

if __name__ == "__main__":
    test()