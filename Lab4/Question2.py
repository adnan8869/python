# Implement Queue using Python

class Queue:
    def __init__(self):
        self.queue = []
        print("Queue Created")
    
    def enqueue(self, data):
        self.queue.append(data)
        print(data,"Enqueued into Queue")
    
    def dequeue(self):
        if len(self.queue) == 0:
            return "Queue is Empty"
        else:
            print(self.queue[0],"Dequeued from Queue")
            return self.queue.pop(0)
    
    def peek(self):
        if len(self.queue) == 0:
            return "Queue is Empty"
        else:
            print(self.queue[0] , "is the Front Element")
            return self.queue[0]
    
    def display(self):
        if len(self.queue) == 0:
            return "Queue is Empty"
        else:
            return self.queue
        
        
queue = Queue()
queue.enqueue(10)
queue.enqueue(20)
queue.enqueue(30)
queue.enqueue(40)
queue.enqueue(50)
queue.dequeue()
queue.peek()
queue.enqueue(60)
queue.dequeue()
print(queue.display())
