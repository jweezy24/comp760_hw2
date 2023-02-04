import numpy as np
from data_parser import *


def calculate_entropy(data):
    probs = [0,0]
    n = len(data)
    for x1,x2,y in data:
        probs[y] += 1
    
    if n == 0:
        probs = [1,1]
    else:
        probs[0]/=n
        probs[1]/=n
    
    if probs[0] > 0 and probs[1] > 0:
        entropy = probs[0]*np.log2(probs[0]) + probs[1]*np.log2(probs[1])
    elif probs[0] == 0 and probs[1] > 0:
        entropy = 0 + probs[1]*np.log2(probs[1])
    elif probs[1] == 0 and probs[0] > 0:
        entropy = probs[0]*np.log2(probs[0]) + 0
    elif probs[0] == 0 and probs[1] == 0:
        entropy = 0
    else:
        raise("Negative value encountered")
        
    return -entropy

def information_gain(parent,left,right):
    E_parent = calculate_entropy(parent)
    E_right = calculate_entropy(right)
    E_left = calculate_entropy(left)

    E_split = (len(right)/len(parent))*E_right + (len(left)/len(parent))*E_left

    
    gain = E_parent - E_split
    
    print(f"left = {left}\t right = {right}")
    split_info = - ((len(right)/len(parent))*np.log2(len(right)/len(parent)) + (len(left)/len(parent))*np.log2(len(left)/len(parent))) 

    print(f"{gain}/{split_info}")
    ratio = gain/split_info
    return ratio 

def find_split(data,feature=0,test=False):
    max_igr = 0
    root_pt = (0,0,-1)
    l=[]
    r=[]
    for i in range(len(data)):
        p = data[i]
        left = []
        right = []
        for x1,x2,y in data:
            if x1 == p[0] and x2 == p[1]:
                continue

            if feature == 0 and x1 <= p[feature]:
                left.append( (x1,x2,y) )
            elif feature == 1 and x2 <= p[feature]:
                left.append( (x1,x2,y) )
            elif feature == 0 and x1 > p[feature]:
                right.append( (x1,x2,y) )
            elif feature == 1 and x2 > p[feature]:
                right.append( (x1,x2,y) )
        
        if len(left) == 0 or len(right) == 0:
            continue
        
        tmp_igr = information_gain(data,left,right)
        print(tmp_igr)
        if test:
            if tmp_igr > max_igr and p[1] == 2:
                max_igr = tmp_igr
                root_pt = p
                l=left
                r=right
        else:
            if tmp_igr > max_igr:
                max_igr = tmp_igr
                root_pt = p
                l=left
                r=right
        
    return root_pt,max_igr,l,r

def iterate_tree(root,level=0):
    ret = "\t"*level+str(root.val)+"\n"
    if root.left != None:
        ret+=iterate_tree(root.left, level+1 )
    else:
        ret+="\t"*(level+1)+f"Left x_{root.dim} >= {root.val[root.dim]} LEAF:{root.val[-1]}"+"\n"
    
    if root.right != None:
        ret+=iterate_tree(root.right, level+1 )
    else:
        if root.val[-1] == 0:
            ret+="\t"*(level+1)+f"Right x_{root.dim} < {root.val[root.dim]} LEAF:1"+"\n"
        else:
            ret+="\t"*(level+1)+f"Right x_{root.dim} < {root.val[root.dim]} LEAF:0"+"\n"
    
    return ret

class Node():
    def __init__(self,val,dim):
        self.val = val
        self.dim = dim
        self.left = None
        self.right = None
        

    def set_left(self,n):
        self.left = n
    
    def set_right(self,n):
        self.right = n

    def get_right(self):
        return self.right
    
    def get_left(self):
        return self.left

    def predict(self):
        return self.val[-1]

class DecisionTree():
    def __init__(self,data,name=""):
        self.data = data

        root_node_0,igr,l,r =  find_split(data,feature=0,test=True)
        root_node_1,igr2,l2,r2 =  find_split(data,feature=1,test=True)

        if igr2 >= igr:
            self.root = Node(root_node_1,1)
            self.build_tree(self.root,l2,r2)
        else:
            self.root = Node(root_node_0,0)
            self.build_tree(self.root,l,r)

        if name != "":
            print(f"Finished with {name}")


    def build_tree(self,root,l,r):
        print("building")
        if len(l) == 0:
            ent_l = 0
        else:
            ent_l = calculate_entropy(l)

        if len(r) == 0:
            ent_r = 0
        else:
            ent_r = calculate_entropy(l)

        if ent_l == 0:
            root.set_left(None)
        else:
            tmp_root,igr,l_tmp,r_tmp = find_split(l,feature=0)
            tmp_root2,igr2,l_tmp2,r_tmp2 = find_split(l,feature=1)
            if igr == 0 and igr2 == 0:
                root.set_left(None)
            if igr > igr2:
                l_n = Node(tmp_root,0)
                self.build_tree(l_n,l_tmp,r_tmp)
                root.set_left(l_n)
            else:
                l_n = Node(tmp_root2,1)
                self.build_tree(l_n,l_tmp2,r_tmp2)
                root.set_left(l_n)

        if ent_r == 0:
            root.set_right(None)
        else:
            tmp_root,igr,l_tmp,r_tmp = find_split(r,feature=0)
            tmp_root2,igr2,l_tmp2,r_tmp2 = find_split(r,feature=1)
            if igr == 0 and igr2 == 0:
                root.set_right(None)
            if igr > igr2:
                r_n = Node(tmp_root,0)
                self.build_tree(r_n,l_tmp,r_tmp)
                root.set_right(r_n)
            else:
                r_n = Node(tmp_root2,1)
                self.build_tree(r_n,l_tmp2,r_tmp2)
                root.set_right(r_n)

        



if __name__ == "__main__":
    path = "data"
    dataset = walk_all_files(path)
    for key in dataset.keys():
        if "bad" in key:
            t = DecisionTree(dataset[key],name=key)
            print(iterate_tree(t.root))