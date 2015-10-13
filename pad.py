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

    def removeDuplicates(self):
        current = self.head
        duplicates = {}
        previous = None

        while current is not None:
            if current.data in duplicates:
                previous.next = current.next
            else:
                duplicates[current.data] = True
                previous = current
            current = current.next

    def kthToLast(self, k):
        n1 = self.head
        n2 = self.head

        for i in range(k - 1):
            if n2 is None:
                return None
            n2 = n2.next

        if n2 is None:
            return None

        while n2.next is not None:
            n1 = n1.next
            n2 = n2.next
        return n1

s = SinglyLinkedList()
s.append(1)
s.append(4)
s.append(5)
