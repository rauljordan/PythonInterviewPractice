from itertools import permutations
"""
With the solution represented as a vector with one queen in each row, we don't have to check to see if two queens are on the same row. By using a permutation generator, we know that no value in the vector is repeated, so we don't have to check to see if two queens are on the same column. Since rook moves don't need to be checked, we only need to check bishop moves.

The technique for checking the diagonals is to add or subtract the column number from each entry, so any two entries on the same diagonal will have the same value (in other words, the sum or difference is unique for each diagnonal). Now all we have to do is make sure that the diagonals for each of the eight queens are distinct. So, we put them in a set (which eliminates duplicates) and check that the set length is eight (no duplicates were removed)."""
def board(vec):
    for col in vec:
        s = ['-']*len(vec)
        s[col] = 'Q'
        print ''.join(s)
    print

n = 8
cols = range(n)
for vec in permutations(cols):
    if (n == len(set(vec[i]+1 for i in cols)) == len(set(vec[i]-1 for i in cols))):
        board(vec)


"""
Pacman DFS
"""
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    start_node = (problem.getStartState(), [], 0)
    explored = []
    frontier = util.Queue()
    frontier.push(start_node)

    if problem.isGoalState(problem.getStartState()):
        return []

    while not frontier.isEmpty():
        (current_state, actions, costs) = frontier.pop()
        if current_state not in explored:
            explored.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            for child_node in problem.getSuccessors(current_state):
                next_state = child_node[0]
                next_action = child_node[1]
                next_cost = child_node[2]

                next_node = (next_state, actions + [next_action], costs + next_cost)
                frontier.push(next_node)
    return []
