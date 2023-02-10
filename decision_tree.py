import numpy as np
from data_parser import *
import matplotlib.pyplot as plt

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

    # print(probs)
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

def parent_child_ratio(child,parent):
    if len(parent) == 0:
        return 1
    else:
        return len(child)/len(parent)

def information_gain(parent,left,right,feature,event):
    E_parent = calculate_entropy(parent)
    E_right = calculate_entropy(right)
    E_left = calculate_entropy(left)

    rp = parent_child_ratio(right,parent)
    lp = parent_child_ratio(left,parent)

    E_split = (rp)*E_right + (lp)*E_left

    
    gain = E_parent - E_split
    
    # print(f"Entropies:\n Parent\t{E_parent}\tLeft\t{E_left}\tRight{E_right}")
    # print(f"USING FEATURE {feature} left = {left}\t right = {right}")
    if lp == 0:
        lp=1
    if rp == 0:
        rp=1

    occs = 0
    for p in parent:
        if p[0] == event[0] and p[1] == event[1] and p[2] == event[2]:
            occs+=1

    entropy_of_event = - (occs/len(parent) * np.log2(occs/len(parent))) 

    ratio = gain/entropy_of_event

    # print(f"INFORMATION GAIN RATIO: {ratio}\tUSING FEATURE {feature}\t left = {left}\t right = {right}")
    return ratio 

def find_split(data,force_node=False,force_ind=0):
    max_igr = 0
    root_pt = (0,0,-1)
    l=[]
    r=[]

    if force_node:
        left = []
        right = []
        p = data[force_ind]
        for k in range(0,2):
            left = []
            right = []

            for j,d in enumerate(data):
                x1,x2,y = d
                if j == force_node:
                    continue

                
                if k == 0 and x1 >= p[k]:
                    left.append( (x1,x2,y) )
                elif k == 1 and x2 >= p[k]:
                    left.append( (x1,x2,y) )
                elif k == 0 and x1 < p[k]:
                    right.append( (x1,x2,y) )
                elif k == 1 and x2 < p[k]:
                    right.append( (x1,x2,y) )
            
            tmp_igr = information_gain(data,left,right,k,p)
    
            if tmp_igr > max_igr:
                max_igr = tmp_igr
                root_pt = p
                l=left
                r=right
                feat = k

        l = left
        r = right
        root_pt = p
        max_igr = information_gain(data,left,right,feat,p)
        return root_pt,max_igr,l,r,feat

    for i in range(len(data)):
        p = data[i]
        left = []
        right = []
        feat = 0
        for k in range(0,2):
            left = []
            right = []

            for j,d in enumerate(data):
                x1,x2,y = d
                # if i == j:
                #     continue

                if k == 0 and x1 >= p[k]:
                    left.append( (x1,x2,y) )
                elif k == 1 and x2 >= p[k]:
                    left.append( (x1,x2,y) )
                elif k == 0 and x1 < p[k]:
                    right.append( (x1,x2,y) )
                elif k == 1 and x2 < p[k]:
                    right.append( (x1,x2,y) )

            
            # if len(left) == 0 or len(right) == 0:
            #     continue
            
            
            tmp_igr = information_gain(data,left,right,k,p)
            # print(tmp_igr,k)

            if tmp_igr > max_igr:
                max_igr = tmp_igr
                root_pt = p
                l=left
                r=right
                feat = k
    
    return root_pt,max_igr,l,r,k

def iterate_tree(root,level=0):
    ret = "\t"*level+str(root.val)+"\n"
    ret+="\t"*(level)+f"Condition x_{root.dim} >= {root.val[root.dim]}"+"\n"
    if root.left != None:
        ret+=iterate_tree(root.left, level+1 )
    else:
        ret+="\t"*(level+1)+f"Left x_{root.dim} >= {root.val[root.dim]} LEAF:{root.predict()}"+"\n"
    
    if root.right != None:
        ret+=iterate_tree(root.right, level+1 )
    else:
        pred = abs(root.predict()-1)
        ret+="\t"*(level+1)+f"Right x_{root.dim} < {root.val[root.dim]} LEAF:{pred}"+"\n"
        
    
    return ret

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

    def predict(self):
        vals = [0,0]
        for x1,x2,y in self.pool:
            vals[y] +=1

        #vals[self.val[-1]]+=1
        print(vals,self.pool)
        if vals[0] > vals[1]:
            return 0
        elif vals[0] < vals[1]:
            return 1
        else:
            return 1

class DecisionTree():
    def __init__(self,data,name="",force_choice=False,choice_ind=0):
        self.data = data

        if force_choice:
            root_node_0,igr,l,r,k =  find_split(data,force_node=True,force_ind=choice_ind)
        else:
            root_node_0,igr,l,r,k =  find_split(data)

        print(f"IGR: {igr} {l}{r}")

        self.root = Node(root_node_0,k,data)
        self.build_tree(self.root,l,r)

        if name != "":
            print(f"Finished with {name}")


    def build_tree(self,root,l,r):
        
        stop_l = False
        stop_r = False
        if len(r) == 0:
            root.set_right(None)
            stop_r = True
        if calculate_entropy(r) == 0:    
            root.set_right(None)
            stop_r = True
        
        if len(l) == 0:
            root.set_left(None)
            stop_l = True
        if calculate_entropy(l) == 0:
            root.set_left(None)
            stop_l = True

        # print(f"SPLITS\tLEFT:{l}\nRIGHT:{r}\n{stop_r},{stop_l}")
        if not stop_l:
            tmp_root,igr,l_tmp,r_tmp,k = find_split(l)
            
            # print(f" Information Gain Ratio: {igr} for left node")
            if igr == 0:
                root.set_left(None)
            else:
                l_n = Node(tmp_root,k,l)
                self.build_tree(l_n,l_tmp,r_tmp)
                root.set_left(l_n)
        
            
        if not stop_r:
            tmp_root,igr,l_tmp,r_tmp,k = find_split(r)

            
            print(f" Information Gain Ratio: {igr} for right node")
            if igr == 0:
                root.set_right(None)
            else:
                r_n = Node(tmp_root,k,r)
                self.build_tree(r_n,l_tmp,r_tmp)
                root.set_right(r_n)
            

    def predict(self,val,node=None):
        if node == None:
            node = self.root
            is_root = True
        else:
            is_root = False



        if val[node.dim] >= node.val[node.dim]:
            if node.left == None:
                return node.predict()
            else:
                return self.predict(val,node=node.left)
        else:
            if node.right == None:
                
                if node.predict() == 1:
                    return 0
                else:
                    return 1
            else:
                return self.predict(val,node=node.right)

def visualize_dataset(ds):
    
    
    labels = ["red","blue"]
    for x1,x2,y in ds:
        plt.scatter(x1,x2,c=labels[y])
    
    plt.show()


if __name__ == "__main__":
    path = "../data"
    dataset = walk_all_files(path)
    for key in dataset.keys():
        if "bad" in key:
            t = DecisionTree(dataset[key],name=key,force_choice=False,choice_ind=8)
            print(iterate_tree(t.root))
            ds = dataset[key]
            error = 0
            X = []
            Y = []
            for d in ds:
                pred = t.predict(d)
                X.append((d[0],d[1]))
                Y.append(d[2])
                if pred != d[2]:
                    error+=1
                    print(pred,d)
            loss = error/len(ds)
            print(f"LOSS = {loss}")
            # visualize_dataset(ds)
            from sklearn import tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X,Y)
            error = 0
            for d in ds:
                x = (d[0],d[1])
                p = clf.predict([x])
                if p[0] != d[2]:
                    error+=1
            loss = error/len(ds)
            print(f"SKLOSS = {loss}")
            text_representation = tree.export_text(clf)
            print(text_representation)