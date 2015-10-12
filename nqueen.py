from itertools import permutations

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
