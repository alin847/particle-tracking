class Node():
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList():
    def __init__(self):
        self.head = None
        self.tail = None


    def __str__(self):
        nodes = []
        curr = self.head
        while curr:
            nodes.append(str(curr.value))
            curr = curr.next
        return " -> ".join(nodes) + " -> None"


    def append(self, value):
        """appends a value to the end of the linked list"""
        node = Node(value)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node


    def extend(self, linked_list):
        """extends the orignal linked list with another linked list"""
        if not self.head:
            self.head = linked_list.head
            self.tail = linked_list.tail
        else:
            self.tail.next = linked_list.head
            self.tail = linked_list.tail


    def get_head(self):
        return self.head.value if self.head else None


    def get_tail(self):
        return self.tail.value if self.tail else None


    def to_list(self):
        """converts linked list to a regular python list"""
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result
