
class Node():
    def __init__(self,val,dim,l):
        self.val = val
        self.dim = dim
        self.pool = l
        self.left = None
        self.right = None

    def set_left(self,n):
        self.left = n
    
    def set_right(self,n):
        self.right = n

    def get_condition(self):
        if self.val != None:
            print(f"Root:{self.val}\n Left x_{self.dim} >= {self.val[self.dim]}  \t Right x_{self.dim} < {self.val[self.dim]} ")
        else:
            print(f"LEAF NODE WITH POOL{self.pool}")

    def predict(self):
        vals = [0,0]
        for x1,x2,y in self.pool:
            vals[y] +=1

        #vals[self.val[-1]]+=1
        
        if vals[0] > vals[1]:
            return 0
        elif vals[0] < vals[1]:
            return 1
        else:
            return 1

