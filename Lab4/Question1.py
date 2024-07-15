# Implement Stack Using Pyhton

class Stack:
    def __init__(self):
        self.stack = []
        print("Stack Created")
    
    def push(self, data):
        self.stack.append(data)
        print(data,"Pushed into Stack")
    
    def pop(self):
        if len(self.stack) == 0:
            return "Stack is Empty"
        else:
            print(self.stack[-1],"Popped from Stack")
            return self.stack.pop()
    
    def peek(self):
        if len(self.stack) == 0:
            return "Stack is Empty"
        else:
            print(self.stack[-1] , "is the Top Element")
            return self.stack[-1]
        
    def display(self):
        if len(self.stack) == 0:
            return "Stack is Empty"
        else:
            return self.stack[::-1]
        
        
        
stack = Stack()
stack.push(10)
stack.push(20)
stack.push(30)
stack.push(40)
stack.push(50)
stack.pop()
stack.peek()
print(stack.display())

# where self is the instance of the class. It is used to access variables that belongs to the class.
# The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.
# means in other words we can say that the variable stack is indirectly used as self.stack
# we use self.stack[-1] to access the last element of the stack because stack is a list and we can access the last element of the list by using -1 index

