"""
Learning every data structure inside out no matter what.
We will review
- lists
- stacks/queues
- hashset/hashmap/hashtable/dictionary
- tree/binary tree
- Heap
- graphs
"""

"""
All Sorting Algorithms
"""
def mergesort(arr):
    """ perform mergesort on a list of numbers

    >>> mergesort([5, 4, 1, 6, 2, 3, 9, 7])
    [1, 2, 3, 4, 5, 6, 7, 9]

    >>> mergesort([3, 2, 4, 2, 1])
    [1, 2, 2, 3, 4]
    """
    n = len(arr)
    # base case, if it just has one or no elements, return it
    if n <= 1: return arr
    a1 = mergesort(arr[:n/2])
    a2 = mergesort(arr[n/2:])
    return merge(a1, a2)

def merge(arr_a, arr_b):
    arr_c = []
    i, j = (0, 0)
    while i < len(arr_a) and j < len(arr_b):
        if arr_a[i] <= arr_b[j]:
            arr_c.append(arr_a[i])
            i += 1
        else:
            arr_c.append(arr_b[j])
            j += 1
    if arr_a[i:]: arr_c.extend(arr_a[i:])
    if arr_b[j:]: arr_c.extend(arr_b[j:])
    return arr_c

def quicksort(a):
    """ quicksort implementation in python
    NOTE: This algo uses O(n) extra space
    to compute quicksort.

    >>> quicksort([6, 4, 8, 2, 1, 9, 10])
    [1, 2, 4, 6, 8, 9, 10]
    """
    n = len(a)
    if n<=1:
        return a
    else:
        from random import randrange
        # finds a pivot by popping from a randrange
        pivot = a.pop(randrange(n))
        # gets lesser and greater and adds them together. Similar to
        # the haskell implemena
        lesser = quicksort([x for x in a if x < pivot])
        greater = quicksort([x for x in a if x >= pivot])
        return lesser + [pivot] + greater


def selectionsort(a):
    """ selectionsort implementation

    >>> selectionsort([6, 4, 8, 2, 1, 9, 10])
    [1, 2, 4, 6, 8, 9, 10]
    """
    for i in range(len(a)):
        min = i
        for j in range(i,len(a)):
            if a[j] < a[min]:
                min = j
        a[i],a[min] = a[min], a[i]
    return a

def bubblesort(a):
    """ bubble sort implementation

    >>> bubblesort([6, 4, 8, 2, 1, 9, 10])
    [1, 2, 4, 6, 8, 9, 10]
    """
    for i in range(len(a)):
        for j in range(i, len(a)):
            if a[i] > a[j]:
                a[i], a[j] = a[j], a[i]
    return a


def insertionsort(a):
    """ insertion sort implementation
    >>> insertionsort([6, 4, 8, 2, 1, 9, 10])
    [1, 2, 4, 6, 8, 9, 10]
    """
    for i in range(len(a)):
        item = a[i]
        j = i
        while j > 0 and a[j-1] > item:
            a[j] = a[j-1]
            j -= 1
        a[j] = item
    return a

from random import randint
def qsort(a, start, end):
    """ quicksort in O(nlogn) and no extra
    memory. In place implementation
    >>> from random import sample
    >>> rand_list = [sample(range(100), 10) for j in range(10)]
    >>> sortedresult = [sorted(r) for r in rand_list]
    >>> for r in rand_list: qsort(r, 0, len(r)-1)
    >>> result = [sortedresult[i] == rand_list[i] for i in range(len(rand_list))]
    >>> print sum(result)
    10
    """
    if start < end:
        p = choosepivot(start, end)
        if p != start:
            a[p], a[start] = a[start], a[p]
        equal = partition(a, start, end)
        qsort(a, start, equal-1)
        qsort(a, equal+1, end)

def partition(a, l, r):
    """ partition array with pivot at a[0]
    in the array a[l...r] that returns the
    index of pivot element
    """
    pivot, i = a[l], l+1
    for j in range(l+1, r+1):
        if a[j] <= pivot:
            a[i],a[j] = a[j],a[i]
            i += 1
    # swap pivot to its correct place
    a[l], a[i-1] = a[i-1], a[l]
    return i-1

def choosepivot(s, e):
    return randint(s,e)

"""
Linked Lists
- Why? arrays are inefficient because they have fixed size. inserting elements
    at the front can be expensive because you have to shift others
- Advantages
    does not have to store all data continuously. dynamic size.
- Disadvantages
    sequential access
- key algorithms
    1) insertion/deletion O(1) at the beginning
    2) indexing O(n)
    3) insert/delete in middle is search time + O(1)
"""
class Node(object):
    def __init__(self, data, next, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

class SinglyLinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None
    def append(self, data):
        n = Node(data, None)
        if self.head is None:
            self.head = self.tail = n
        else:
            self.tail.next = n
        self.tail = n

    def remove(self, data):
        current = self.head
        previous = None
        while current is not None:
            if current.data == data:
                if previous is not None:
                    previous.next = current.next
                else:
                    self.head = current.next
            previous = current
            current = current.next

    def show(self):
        current = self.head
        while current is not None:
            print current.data, "->",
            current = current.next
        print None




class DoublyLinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, data):
        n = Node(data, None, None)
        if self.head is None:
            self.head = self.tail = n
        else:
            self.tail.next = n
        self.tail = n

    def show(self):
        current = self.head
        while current is not None:
            print current.data,"->",
            current = current.next
        print None

    def remove(self, data):
        current = self.head
        while current is not None:
            if current.data == data:
                if current.prev is not None:
                    current.prev.next = current.next
                    current.next.prev = current.prev
                else:
                    self.head = current.next
                    current.next.prev = None
            current = current.next

"""
Stacks and Queue
- Implement a FIFO and LIFO
"""
class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

    def peek(self):
        if self.isEmpty(self.list):
            return None
        else:
            return self.list[-1]

    def __str__(self):
        return str(self.list)


"""
to implemenet sort, we need to use another one. Pop from s1 and store in temp
variable and then search the real stack by popping until you find the right place.
Then we add it to the real stack and we repeat the process
"""

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

"""
Hashset
- Implement a hashset in python
"""
class HashSet(object):
    def __init__(self, elements = []):
        self._elements = {}
        for e in elements:
            self._elements[e] = 1

    def add(self, item):
        self._elements[item] = 1

    def union(self, s):
        pass

"""
Binary Tree

1) Properties
    - n = 2h + 1 is at least the number of nodes
    - height is log n
    - Searching is O(h). since binary trees can degenerate to a linked list
"""
class Node(object):
    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right

class BinaryTree(object):

    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value, None, None)
        else:
            self.recursive_insert(value, self.root)

    def recursive_insert(self, value, node):
        if value < node.data:
            if node.left is None:
                node.left = Node(value, None, None)
            else:
                self.recursive_insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value, None, None)
            else:
                self.recursive_insert(value, node.right)

    def isBalanced(self):
        return self.isBalancedRecursive(self.root)

    def isBalancedRecursive(self, node):
        if node.left is not None and node.right is not None:
            return self.isBalancedRecursive(node.left) and self.isBalancedRecursive(node.right)
        if node.left is None and node.right is None:
            return True
        if node.left is None and node.right is not None:
            return False
        if node.left is not None and node.right is None:
            return False

    def getHeightRecursive(self, node, depth):

        if node.left is None and node.right is None:
            return depth
        if node.left is None and node.right is not None:
            depth = depth + 1
            return self.getHeightRecursive(node.right, depth)
        if node.left is not None and node.right is None:
            depth = depth + 1
            return self.getHeightRecursive(node.left, depth)
        else:
            depth = depth + 1
            return self.getHeightRecursive(node.left, depth) + self.getHeightRecursive(node.right, depth)

    def getHeight(self):
        return max([self.getHeightRecursive(self.root.left, 0),
                        self.getHeightRecursive(self.root.right, 0)])

    def printTree(self):
        if(self.root != None):
            self._printTree(self.root)

    def _printTree(self, node):
        if(node != None):
            self._printTree(node.left)
            print str(node.data) + ' '
            self._printTree(node.right)

    def bfs(self):
        frontier = Queue()
        explored = []
        explored.append(self.root)
        frontier.push(self.root)

        # the list of linked lists at each depth
        rootList = SinglyLinkedList()
        rootList.append(self.root.data)
        lists = [rootList]

        while not frontier.isEmpty():
            node = frontier.pop()
            children = [node.left, node.right]
            child_list = SinglyLinkedList()

            if node not in explored:
                for child in children:
                    child_list.append(child)
                    lists.append(child_list)
                    frontier.push(child)
        return lists

b = BinaryTree()
b.insert(5)
b.insert(3)
b.insert(6)
print b.bfs()
"""
Heap
Why? Sorted by keys. Essentially we can create a min heap or
max heap from this
"""
import math
class minheap(object):
    """
    Heap class - made of keys and items
    methods: build_heap, heappush, heappop
    """

    MIN_HEAP = True

    def __init__(self, nums=None):
        self.heap = []
        if nums:
            self.build_heap(nums)

    def __str__(self):
        return "Min-heap with %s items" % (len(self.heap))

    def max_elements(self):
        return len(self.heap)

    def height(self):
        return math.ceil(math.log(len(self.heap)) / math.log(2))

    def is_leaf(self, i):
        """ returns True if i is a leaf node """
        return i > int(math.ceil((len(self.heap) - 2) / 2.0))

    def parent(self, i):
        if i == 0:
            return -1
        elif i % 2 != 0: # odd
            return (i - 1) / 2
        return (i - 2) / 2

    def leftchild(self, i):
        return 2 * i + 1

    def rightchild(self, i):
        return 2 * i + 2

    def heapify(self, i):
        l = self.leftchild(i)
        r = self.rightchild(i)
        smallest = i
        if l < self.max_elements() and self.heap[l] < self.heap[smallest]:
            smallest = l
        if r < self.max_elements() and self.heap[r] < self.heap[smallest]:
            smallest = r
        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self.heapify(smallest)

    def build_heap(self, elem):
        """ transforms a list of elements into a heap
        in linear time """
        self.heap = elem[:]
        last_leaf = self.parent(len(self.heap))
        for i in range(last_leaf, -1, -1):
            self.heapify(i)

    def heappush(self, x):
        """ Adds a new item x in the heap"""
        i = len(self.heap)
        self.heap.append(x)
        parent = self.parent(i)
        while parent != -1 and self.heap[i] < self.heap[parent]:
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            i = parent
            parent = self.parent(i)

    def heappop(self):
        """ extracts the root of the heap, min or max
        depending on the kind of heap"""
        if self.max_elements():
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            pop = self.heap.pop()
            self.heapify(0)
            return pop
        raise Exception("Heap is empty")

"""
Sieve of Erathosthenes
"""

def generate_primes(n):
    bool_array = [False, False] + [True] * n              # start with all values as True, except 0 and 1
    for i in range(2, int(ceil(sqrt(n)))):                # only go to till square root of n
        if bool_array[i]:                                 # if the number is marked as prime
            for j in range(i*i,n+1,i):                    # iterate through all its multiples
                bool_array[j] = False                     # and mark them as False
    primes = [i for i in range(n+1) if bool_array[i]]     # return all numbers which are marked as True
    return primes

"""
run length encoding
"""
from itertools import groupby
def encode(l):
	return [(len(list(group)),name) for name, group in groupby(l)]

def decode(l):
	return sum([length * [item] for length,item in l],[])

"""
Tower of Hanoi Using Recursion
"""


class Graph(object):
    """
    Graph class - made of nodes and edges

    methods: add_edge, add_edges, add_node, add_nodes, has_node,
    has_edge, nodes, edges, neighbors, del_node, del_edge, node_order,
    set_edge_weight, get_edge_weight
    """

    DEFAULT_WEIGHT = 1
    DIRECTED = False

    def __init__(self):
        self.node_neighbors = {}

    def __str__(self):
        return "Undirected Graph \nNodes: %s \nEdges: %s" % (self.nodes(), self.edges())

    def add_nodes(self, nodes):
        """
        Takes a list of nodes as input and adds these to a graph
        """
        for node in nodes:
            self.add_node(node)

    def add_node(self, node):
        """
        Adds a node to the graph
        """
        if node not in self.node_neighbors:
            self.node_neighbors[node] = {}
        else:
            raise Exception("Node %s is already in graph" % node)

    def has_node(self, node):
        """
        Returns boolean to indicate whether a node exists in the graph
        """
        return node in self.node_neighbors

    def add_edge(self, edge, wt=DEFAULT_WEIGHT, label=""):
        """
        Add an edge to the graph connecting two nodes.
        An edge, here, is a pair of node like C(m, n) or a tuple
        """
        u, v = edge
        if (v not in self.node_neighbors[u] and u not in self.node_neighbors[v]):
            self.node_neighbors[u][v] = wt
            if (u!=v):
                self.node_neighbors[v][u] = wt
        else:
            raise Exception("Edge (%s, %s) already added in the graph" % (u, v))

    def add_edges(self, edges):
        """ Adds multiple edges in one go. Edges, here, is a list of
        tuples"""
        for edge in edges:
            self.add_edge(edge)

    def nodes(self):
        """
        Returns a list of nodes in the graph
        """
        return self.node_neighbors.keys()

    def has_edge(self, edge):
        """
        Returns a boolean to indicate whether an edge exists in the
        graph. An edge, here, is a pair of node like C(m, n) or a tuple
        """
        u, v = edge
        return v in self.node_neighbors.get(u, [])

    def neighbors(self, node):
        """
        Returns a list of neighbors for a node
        """
        if not self.has_node(node):
            raise "Node %s not in graph" % node
        return self.node_neighbors[node].keys()

    def del_node(self, node):
        """
        Deletes a node from a graph
        """
        for each in list(self.neighbors(node)):
            if (each != node):
                self.del_edge((each, node))
        del(self.node_neighbors[node])

    def del_edge(self, edge):
        """
        Deletes an edge from a graph. An edge, here, is a pair like
        C(m,n) or a tuple
        """
        u, v = edge
        if not self.has_edge(edge):
            raise Exception("Edge (%s, %s) not an existing edge" % (u, v))
        del self.node_neighbors[u][v]
        if (u!=v):
            del self.node_neighbors[v][u]

    def node_order(self, node):
        """
        Return the order or degree of a node
        """
        return len(self.neighbors(node))


    def edges(self):
        """
        Returns a list of edges in the graph
        """
        edge_list = []
        for node in self.nodes():
            edges = [(node, each) for each in self.neighbors(node)]
            edge_list.extend(edges)
        return edge_list

    # Methods for setting properties on nodes and edges
    def set_edge_weight(self, edge, wt):
        """Set the weight of the edge """
        u, v = edge
        if not self.has_edge(edge):
            raise Exception("Edge (%s, %s) not an existing edge" % (u, v))
        self.node_neighbors[u][v] = wt
        if u != v:
            self.node_neighbors[v][u] = wt

    def get_edge_weight(self, edge):
        """Returns the weight of an edge """
        u, v = edge
        if not self.has_edge((u, v)):
            raise Exception("%s not an existing edge" % edge)
        return self.node_neighbors[u].get(v, self.DEFAULT_WEIGHT)

    def get_edge_weights(self):
        """ Returns a list of all edges with their weights """
        edge_list = []
        unique_list = {}
        for u in self.nodes():
            for v in self.neighbors(u):
                if u not in unique_list.get(v, set()):
                    edge_list.append((self.node_neighbors[u][v], (u, v)))
                    unique_list.setdefault(u, set()).add(v)
        return edge_list

from collections import deque
from copy import deepcopy
from union_find.unionfind import UnionFind
import heapq

def BFS(gr, s):
    """ Breadth first search
    Returns a list of nodes that are "findable" from s """
    if not gr.has_node(s):
        raise Exception("Node %s not in graph" % s)
    nodes_explored = set([s])
    q = deque([s])
    while len(q)!=0:
        node = q.popleft()
        for each in gr.neighbors(node):
            if each not in nodes_explored:
                nodes_explored.add(each)
                q.append(each)
    return nodes_explored

def shortest_hops(gr, s):
    """ Finds the shortest number of hops required
    to reach a node from s. Returns a dict with mapping:
    destination node from s -> no. of hops
    """
    if not gr.has_node(s):
        raise Exception("Node %s is not in graph" % s)
    else:
        dist = {}
        q = deque([s])
        nodes_explored = set([s])
        for n in gr.nodes():
            if n == s: dist[n] = 0
            else: dist[n] = float('inf')
        while len(q) != 0:
            node = q.popleft()
            for each in gr.neighbors(node):
                if each not in nodes_explored:
                    nodes_explored.add(each)
                    q.append(each)
                    dist[each] = dist[node] + 1
        return dist

def undirected_connected_components(gr):
    """ Returns a list of connected components
    in an undirected graph """
    if gr.DIRECTED:
        raise Exception("This method works only with a undirected graph")
    explored = set([])
    con_components = []
    for node in gr.nodes():
        if node not in explored:
            reachable_nodes = BFS(gr, node)
            con_components.append(reachable_nodes)
            explored |= reachable_nodes
    return con_components

def DFS(gr, s):
    """ Depth first search wrapper """
    path = set([])
    depth_first_search(gr, s, path)
    return path

def depth_first_search(gr, s, path):
    """ Depth first search
    Returns a list of nodes "findable" from s """
    if s in path: return False
    path.add(s)
    for each in gr.neighbors(s):
        if each not in path:
            depth_first_search(gr, each, path)

def topological_ordering(digr_ori):
    """ Returns a topological ordering for a
    acyclic directed graph """
    if not digr_ori.DIRECTED:
        raise Exception("%s is not a directed graph" % digr)
    digr = deepcopy(digr_ori)
    ordering = []
    n = len(digr.nodes())
    while n > 0:
        sink_node = find_sink_node(digr)
        ordering.append((sink_node, n))
        digr.del_node(sink_node)
        n -= 1
    return ordering

def find_sink_node(digr):
    """ Finds a sink node (node with all incoming arcs)
    in the directed graph. Valid for a acyclic graph only """
    # first node is taken as a default
    node = digr.nodes()[0]
    while digr.neighbors(node):
        node = digr.neighbors(node)[0]
    return node

def directed_connected_components(digr):
    """ Returns a list of strongly connected components
    in a directed graph using Kosaraju's two pass algorithm """
    if not digr.DIRECTED:
        raise Exception("%s is not a directed graph" % digr)
    finishing_times = DFS_loop(digr.get_transpose())
    # use finishing_times in descending order
    nodes_explored, connected_components = [], []
    for node in finishing_times[::-1]:
        component = []
        outer_dfs(digr, node, nodes_explored, component)
        if component:
            nodes_explored += component
            connected_components.append(component)
    return connected_components

def outer_dfs(digr, node, nodes_explored, path):
    if node in path or node in nodes_explored:
        return False
    path.append(node)
    for each in digr.neighbors(node):
        if each not in path or each not in nodes_explored:
            outer_dfs(digr, each, nodes_explored, path)

def DFS_loop(digr):
    """ Core DFS loop used to find strongly connected components
    in a directed graph """
    node_explored = set([]) # list for keeping track of nodes explored
    finishing_times = [] # list for adding nodes based on their finishing times
    for node in digr.nodes():
        if node not in node_explored:
            leader_node = node
            inner_DFS(digr, node, node_explored, finishing_times)
    return finishing_times

def inner_DFS(digr, node, node_explored, finishing_times):
    """ Inner DFS used in DFS loop method """
    node_explored.add(node) # mark explored
    for each in digr.neighbors(node):
        if each not in node_explored:
            inner_DFS(digr, each, node_explored, finishing_times)
    global finishing_counter
    # adds nodes based on increasing order of finishing times
    finishing_times.append(node)

def shortest_path(digr, s):
    """ Finds the shortest path from s to every other vertex findable
    from s using Dijkstra's algorithm in O(mlogn) time. Uses heaps
    for super fast implementation """
    nodes_explored = set([s])
    nodes_unexplored = DFS(digr, s) # all accessible nodes from s
    nodes_unexplored.remove(s)
    dist = {s:0}
    node_heap = []

    for n in nodes_unexplored:
        min = compute_min_dist(digr, n, nodes_explored, dist)
        heapq.heappush(node_heap, (min, n))

    while len(node_heap) > 0:
        min_dist, nearest_node = heapq.heappop(node_heap)
        dist[nearest_node] = min_dist
        nodes_explored.add(nearest_node)
        nodes_unexplored.remove(nearest_node)

        # recompute keys for just popped node
        for v in digr.neighbors(nearest_node):
            if v in nodes_unexplored:
                for i in range(len(node_heap)):
                    if node_heap[i][1] == v:
                        node_heap[i] = (compute_min_dist(digr, v, nodes_explored, dist), v)
                        heapq.heapify(node_heap)

    return dist
