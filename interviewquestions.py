def binary_search(l, value):
    low = 0
    high = len(l)-1
    while low <= high:
        mid = (low+high)//2
        if l[mid] > value: high = mid-1
        elif l[mid] < value: low = mid+1
        else: return mid
    return -1

#=======================================================================
#  Author: Isai Damier
#  Title: Radix Sort
#  Project: geekviewpoint
#  Package: algorithms
#
#  Statement:
#  Given a disordered list of integers, rearrange them in natural order.
#
#  Sample Input: [18,5,100,3,1,19,6,0,7,4,2]
#
#  Sample Output: [0,1,2,3,4,5,6,7,18,19,100]
#
#  Time Complexity of Solution:
#  Best Case O(kn); Average Case O(kn); Worst Case O(kn),
#  where k is the length of the longest number and n is the
#  size of the input array.
#
#  Note: if k is greater than log(n) then an nlog(n) algorithm would
#  be a better fit. In reality we can always change the radix
#  to make k less than log(n).
#
#  Approach:
#  radix sort, like counting sort and bucket sort, is an integer based
#  algorithm (i.e. the values of the input array are assumed to be
#  integers). Hence radix sort is among the fastest sorting algorithms
#  around, in theory. The particular distinction for radix sort is
#  that it creates a bucket for each cipher (i.e. digit); as such,
#  similar to bucket sort, each bucket in radix sort must be a
#  growable list that may admit different keys.
#
#  For decimal values, the number of buckets is 10, as the decimal
#  system has 10 numerals/cyphers (i.e. 0,1,2,3,4,5,6,7,8,9). Then
#  the keys are continuously sorted by significant digits.
#=======================================================================
 def radixsort( aList ):
  RADIX = 10
  maxLength = False
  tmp , placement = -1, 1

  while not maxLength:
    maxLength = True
    # declare and initialize buckets
    buckets = [list() for _ in range( RADIX )]

    # split aList between lists
    for  i in aList:
      tmp = i / placement
      buckets[tmp % RADIX].append( i )
      if maxLength and tmp > 0:
        maxLength = False

    # empty lists into aList array
    a = 0
    for b in range( RADIX ):
      buck = buckets[b]
      for i in buck:
        aList[a] = i
        a += 1

    # move to next digit
    placement *= RADIX

"""
Find the most frequent integer in an array
"""
def most_freq(xs):
    occurences = {}
    for i in xs:
        if i in occurences.keys():
            occurences[i] += 1
        else:
            occurences[i] = 0
    v = list(occurences.values())
    k = list(occurences.keys())

    print k[v.index(max(v))]

"""
Find pairs in an integer array whose sum is equal to 10
"""

"""
Given 2 integer arrays, determine of the 2nd array is a rotated
version of the 1st array
"""



"""
Write fibbonaci iteratively and recursively
"""
def memoize(fn, arg):
    memo = {}
    if arg not in memo:
        memo[arg] = fn(arg)
        return memo[arg]


def fib(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fib(n - 2) + fib(n - 1)


"""
Given a function rand5() that returns a random int between 0 and 5, implement rand7()

How does it work? Think of it like this: imagine printing out this double-dimension array on paper, tacking it up to a dart board and randomly throwing darts at it. If you hit a non-zero value, it's a statistically random value between 1 and 7, since there are an equal number of non-zero values to choose from. If you hit a zero, just keep throwing the dart until you hit a non-zero. That's what this code is doing: the i and j indexes randomly select a location on the dart board, and if we don't get a good result, we keep throwing darts.
"""
import random
def rand5():
    return random.randint(1,5)

def rand7():
    values = [
        [1,2,3,4,5],
        [6,7,1,2,3],
        [4,5,6,7,1],
        [2,3,4,5,6],
        [7,0,0,0,0]
    ]

    result = 0
    while result == 0:
        i = rand5()
        j = rand5()
        result = values[i - 1][j - 1]
    return result

"""
Newtons method for sq roots
"""
def check(x, guess):
    return (abs(guess*guess - x) < 0.001)

def newton(x, guess):
    while not check(x, guess):
        guess = (guess + (x/guess)) / 2.0
    return guess

""" String Questions"""

"""
Find the first non-repeated character in a String
"""
def first_non_repeated(s):
    repeated = set()
    for letter in s:
        if letter in repeated:
            return letter
        else:
            repeated.add(letter)
"""
Reverse a String iteratively and recursively
"""

def rev_iterative(s):
    reversed = []
    for letter in s:
        reversed.insert(0,letter)
    return ''.join(reversed)

def rev_recursive(s):
    if s == "":
        return s
    else:
        return rev_recursive(s[1:]) + s[0]

"""
Determine if 2 Strings are anagrams
"""
def anagrams(s1, s2):
    bools = []
    for letter1, letter2 in zip(s1, s2):
        if (letter1 in s2) and (letter2 in s1):
            bools.append(True)
        else:
            bools.append(False)
    return False not in bools

"""
Check if String is a palindrome
"""
def palindrome(s):
    return s == rev_recursive(s)

def permute_string(s):
    from itertools import permutations
    string_list = permutations(list(s))
    for item in string_list:
        print ''.join(item)

"""
Implement a Queue using two stacks
"""
class Queue2Stack(object):

    def __init__(self):
        from datastructures import Stack
        self.inbox = Stack()
        self.outbox = Stack()

    def push(self,item):
        self.inbox.push(item)

    def pop(self):
        if self.outbox.isEmpty():
            while not self.inbox.isEmpty():
                self.outbox.push(self.inbox.pop())
        return self.outbox.pop()

"""
Given an image represented by an NxN matrix, write a method to rotate the
image by 90 degrees. Can you do this in place?
"""
def rotate_mat(mat):
    return [list(row) for row in zip(*mat)[::-1]]

print rotate_mat([[1,2],[3,4]])

"""
Pythonic fibbonaci
"""
def fib(n):
   fibValues = [0,1]
   for i in range(2,n+1):
      fibValues.append(fibValues[i-1] + fibValues[i-2])
   return fibValues[n]

"""
Hanoi
"""
def hanoi(n, source, helper, target):
    if n > 0:
        # move tower of size n - 1 to helper:
        hanoi(n - 1, source, target, helper)
        # move disk from source peg to target peg
        if source:
            target.append(source.pop())
        # move tower of size n-1 from helper to target
        hanoi(n - 1, helper, source, target)

#=======================================================================
# Author: Isai Damier
# Title: Longest Increasing Subsequence
# Project: geekviewpoint
# Package: algorithms
#
# Statement:
#   Given a sequence of numbers, find a longest increasing subsequence.
#
#  Time Complexity: O(n^2)
#
# Sample Input: [8,1,2,3,0,5]
# Sample Output: [1,2,3,5]
#
# DEFINITION OF SUBSEQUENCE:
#   A sequence is a particular order in which related objects follow
#   each other (e.g. DNA, Fibonacci). A sub-sequence is a sequence
#   obtained by omitting some of the elements of a larger sequence.
#
#   SEQUENCE       SUBSEQUENCE     OMISSION
#   [3,1,2,5,4]     [1,2]            3,5,4
#   [3,1,2,5,4]     [3,1,4]          2,5
#
#   SEQUENCE       NOT SUBSEQUENCE   REASON
#   [3,1,2,5,4]     [4,2,5]           4 should follow 5
#
# STRATEGY:
#   Illustrating by finding a longest increasing subsequence
#   of [8,1,2,3,0,5]:
#
#   - Start by finding all subsequences of size 1: [8],[1],[2],[3],[0],[5];
#     each element is its own increasing subsequence.
#
#   - Since we already have the solutions for the size 1 subsequences,
#     we can use them in solving for the size two subsequences. For
#     instance, we already know that 0 is the smallest element of an
#     increasing subsequence of size 1, i.e. the subsequence [0].
#     Therefore, all we need to get a subsequence of size 2 is add an
#     element greater than 0 to [0]: [0,5]. The other size 2
#     subsequences are: [1,2], [1,3], [1,5], [2,3], [2,5], [3,5].
#
#   - Now we use the size 2 subsequences to get the size 3 subsequences:
#     [1,2,3], [1,2,5], [1,3,5], [2,3,5]
#
#   - Then we use the size 3 subsequences to get the size 4 subsequences:
#     [1,2,3,5]. Since there are no size 5 solutions, we are done.
#
# SUMMARY:
#   Instead of directly solving the big problem, we solved a smaller
#   version and then 'copied and pasted' the solution of the subproblem
#   to find the solution to the big problem. To make the 'copy and paste'
#   part easy, we use a table (i.e. array) to track the subproblems
#   and their solutions. This strategy as a whole is called Dynamic
#   Programming. The tabling part is known as memoization, which means
#   writing memo.
#
#   To recognize whether you can use dynamic programming on a problem,
#   look for the following two traits: optimal substructures and
#   overlapping subproblems.
#
#   Optimal Substructures: the ability to 'copy and paste' the solution
#     of a subproblem plus an additional trivial amount of work so to
#     solve a larger problem. For example, we were able to use [1,2]
#     itself an optimal solution to the problem [8,1,2] to get [1,2,3]
#     as an optimal solution to the problem [8,1,2,3].
#
#   Overlapping Subproblems: Okay. So in our approach the solution grew
#     from left to right: [1] to [1,2] to [1,2,3] etc. But in reality
#     we could have solved the problem using recursion trees so that
#     for example [1,2] could be reached either from [1] or from [2].
#     That wouldn't really be a problem except we would be solving for
#     [1,2] more than once. Anytime a recursive solution would lead to
#     such overlaps, the bet is dynamic programming is the way to go.
#
#          [1]                 [2]
#         / | \               / | \
#        /  |  \             /  |  \
#       /   |   \           /   |   \
#   [1,2] [1,3] [1,5]   [1,2] [2,3] [2,5]
#
# Dynamic Programming = Optimal Substructures + Overlapping Subproblems
# Divide and Conquer = Optimal Substructures - Overlapping Subproblems
#   see merge sort: http://www.geekviewpoint.com/python/sorting/mergesort
#
# Alternate coding: Not really much difference here, just another code
#   that some readers will find more intuitive:
#
#      m = [1]*len(A)
#
#      for x in range(len(A)):
#        for y in range(x):
#         if m[y] >= m[x] and A[y] < A[x]:
#           m[x]+=1
#
#      max_value = max(m)
#
#      result = []
#      for i in range(m-1,-1,-1):
#        if max == m[i]:
#          result.append(A[i])
#          max-=1
#
#      result.reverse()
#      return result
#=======================================================================
"""
def LIS( A ):m = [0] * len( A ) # m = [1]*len(A) not important here
  for x in range( len( A ) - 2, -1, -1 ):
    for y in range( len( A ) - 1, x, -1 ):
      if A[x] < A[y] and m[x] <= m[y]:
        m[x] += 1 # or use m[x] = m[y] + 1

  #===================================================================
  # Use the following snippet or the one line below to get max_value
  # max_value=m[0]
  # for i in range(m):
  #  if max_value < m[i]:
  #    max_value = m[i]
  #===================================================================
  max_value = max( m )

  result = []
  for i in range( len( m ) ):
    if max_value == m[i]:
      result.append( A[i] )
      max_value -= 1

  return result


"""


def knapsack(items, maxweight):
    """
    Solve the knapsack problem by finding the most valuable
    subsequence of `items` subject that weighs no more than
    `maxweight`.

    `items` is a sequence of pairs `(value, weight)`, where `value` is
    a number and `weight` is a non-negative integer.

    `maxweight` is a non-negative integer.

    Return a pair whose first element is the sum of values in the most
    valuable subsequence, and whose second element is the subsequence.

    >>> items = [(4, 12), (2, 1), (6, 4), (1, 1), (2, 2)]
    >>> knapsack(items, 15)
    (11, [(2, 1), (6, 4), (1, 1), (2, 2)])
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    @memoized
    def bestvalue(i, j):
        if i == 0: return 0
        value, weight = items[i - 1]
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            return max(bestvalue(i - 1, j),
                       bestvalue(i - 1, j - weight) + value)

    j = maxweight
    result = []
    for i in xrange(len(items), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            result.append(items[i - 1])
            j -= items[i - 1][1]
    result.reverse()
    return bestvalue(len(items), maxweight), result

def pascals_triangle(n_rows):
    results = [] # a container to collect the rows
    for _ in range(n_rows):
        row = [1] # a starter 1 in the row
        if results: # then we're in the second row or beyond
            last_row = results[-1] # reference the previous row
            # this is the complicated part, it relies on the fact that zip
            # stops at the shortest iterable, so for the second row, we have
            # nothing in this list comprension, but the third row sums 1 and 1
            # and the fourth row sums in pairs. It's a sliding window.
            row.extend([sum(pair) for pair in zip(last_row, last_row[1:])])
            # finally append the final 1 to the outside
            row.append(1)
        results.append(row) # add the row to the results.
    return results


"""
TSP Problem!
"""

from itertools import permutations


def distance(point1, point2):
    """
    Returns the Euclidean distance of two points in the Cartesian Plane.

    >>> distance([3,4],[0,0])
    5.0
    >>> distance([3,6],[10,6])
    7.0
    """
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** 0.5


def total_distance(points):
    """
    Returns the length of the path passing throught
    all the points in the given order.

    >>> total_distance([[1,2],[4,6]])
    5.0
    >>> total_distance([[3,6],[7,6],[12,6]])
    9.0
    """
    return sum([distance(point, points[index + 1]) for index, point in enumerate(points[:-1])])


def travelling_salesman(points, start=None):
    """
    Finds the shortest route to visit all the cities by bruteforce.
    Time complexity is O(N!), so never use on long lists.

    >>> travelling_salesman([[0,0],[10,0],[6,0]])
    ([0, 0], [6, 0], [10, 0])
    >>> travelling_salesman([[0,0],[6,0],[2,3],[3,7],[0.5,9],[3,5],[9,1]])
    ([0, 0], [6, 0], [9, 1], [2, 3], [3, 5], [3, 7], [0.5, 9])
    """
    if start is None:
        start = points[0]
    return min([perm for perm in permutations(points) if perm[0] == start], key=total_distance)


def optimized_travelling_salesman(points, start=None):
    """
    As solving the problem in the brute force way is too slow,
    this function implements a simple heuristic: always
    go to the nearest city.

    Even if this algoritmh is extremely simple, it works pretty well
    giving a solution only about 25% longer than the optimal one (cit. Wikipedia),
    and runs very fast in O(N^2) time complexity.

    >>> optimized_travelling_salesman([[i,j] for i in range(5) for j in range(5)])
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 4], [1, 3], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 4], [3, 3], [3, 2], [3, 1], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]
    >>> optimized_travelling_salesman([[0,0],[10,0],[6,0]])
    [[0, 0], [6, 0], [10, 0]]
    """
    if start is None:
        start = points[0]
    must_visit = points
    path = [start]
    must_visit.remove(start)
    while must_visit:
        nearest = min(must_visit, key=lambda x: distance(path[-1], x))
        path.append(nearest)
        must_visit.remove(nearest)
    return path


points = [[0, 0], [1, 5.7], [2, 3], [3, 7],
            [0.5, 9], [3, 5], [9, 1], [10, 5]]
print("""The minimum distance to visit all the following points: {}
starting at {} is {}. The optimized algoritmh yields a path long {}.""".format(
        tuple(points),
        points[0],
        total_distance(travelling_salesman(points)),
        total_distance(optimized_travelling_salesman(points))))



def numIslands(grid):
        if not grid: return 0
        m, n, c= len(grid), len(grid[0]), 0
        visited= [[False for i in xrange(n)]for i in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                if not visited[i][j] and grid[i][j]==1:
                    dfs(grid, visited, i, j, m, n)
                    c += 1
        return c

def dfs(grid, visited, i, j, m, n):
    visited[i][j]= True
    if i+1<m and grid[i+1][j]==1 and not visited[i+1][j]:
        dfs(grid, visited, i+1, j, m, n)
    if i-1>=0 and grid[i-1][j]==1 and not visited[i-1][j]:
        dfs(grid, visited, i-1, j, m, n)
    if j+1<n and grid[i][j+1]==1 and not visited[i][j+1]:
        dfs(grid, visited, i, j+1, m, n)
    if j-1>=0 and grid[i][j-1]==1 and not visited[i][j-1]:
        dfs(grid, visited, i, j-1, m, n)


a = [[1,0,1],[0,0,0],[0,0,0]]

print numIslands(a)


_dp = [0]
def numSquares(self, n):
    dp = self._dp
    while len(dp) <= n:
        dp += min(dp[-i*i] for i in range(1, int(len(dp)**0.5+1))) + 1,
    return dp[n]


def klargest(k, xs):
    if k > len(xs):
        raise Exception("k is larger than size of list")
    maximum = None
    while k > 0:
        maximum = max(xs)
        xs = [x for x in xs if x != maximum]
        k -= 1
    return maximum


def minimumEditDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

print(minimumEditDistance("kitten","sitting"))
print(minimumEditDistance("rosettacode","raisethysword"))

"""
Permutations
"""
def permutations(string, step = 0):

    # if we've gotten to the end, print the permutation
    if step == len(string):
        print "".join(string)

    # everything to the right of step has not been swapped yet
    for i in range(step, len(string)):

        # copy the string (store as array)
        string_copy = [character for character in string]

        # swap the current index with the step
        string_copy[step], string_copy[i] = string_copy[i], string_copy[step]

        # recurse on the portion of the string that has not been swapped yet (now it's index will begin with step + 1)
        permutations(string_copy, step + 1)
