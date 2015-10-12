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
b.insert(3)
b.insert(6)
b.insert(5)
b.printTree()
