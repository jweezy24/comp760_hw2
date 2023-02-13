import numpy as np
from data_parser import *
import matplotlib.pyplot as plt

def calculate_entropy(data,condition=2):
    probs = {}
    n = len(data)

    
    for x1,x2,y in data:
        if condition == 0:
            if x1 not in probs.keys():
                probs[x1] = 1
            else:
                probs[x1] += 1
        elif condition == 1:
            if x2 not in probs.keys():
                probs[x2] = 1
            else:
                probs[x2] += 1
        elif condition == 2:
            if y not in probs.keys():
                probs[y] = 1
            else:
                probs[y] += 1

    entropy = 0
    for key in probs.keys():
        p = probs[key]/n
        entropy += (p)*np.log2(p) 
        
    return -entropy

def parent_child_ratio(child,parent):
    if len(parent) == 0:
        return 1
    else:
        return len(child)/len(parent)

def information_gain(parent,left,right,event,feature=2):
    E_parent = calculate_entropy(parent,feature)
    E_right = calculate_entropy(right,feature)
    E_left = calculate_entropy(left,feature)

    rp = parent_child_ratio(right,parent)
    lp = parent_child_ratio(left,parent)

    E_split = (rp)*E_right + (lp)*E_left

    
    gain = E_parent - E_split


    if lp == 0:
        lp=1
    if rp == 0:
        rp=1

    occs = 0
    for p in parent:
        if p[0] == event[0] and p[1] == event[1] and p[2] == event[2]:
            occs+=1

    entropy_of_event = - (occs/len(parent) * np.log2(occs/len(parent))) 

    if entropy_of_event == 0:
        ratio = 0
    else:
        ratio = gain/entropy_of_event

    # print(f"SPLIT CANIDATE = {event}\n IGR = ({E_parent} - {E_split}) / {entropy_of_event} = {ratio}")

    return ratio 

def find_split(data,force_node=False,force_ind=0):
    max_igr = 0
    root_pt = (0,0,-1)
    l=[]
    r=[]
    max_dim = 0

    if force_node:
        left = []
        right = []
        p = data[force_ind]
        for k in range(0,2):
            left = []
            right = []

            for j,d in enumerate(data):
                
                if j == force_node:
                    continue
                
                if d[k] >= p[k]:
                    left.append(d)
                elif d[k] < p[k]:
                    right.append(d)
                
                
            
            tmp_igr = information_gain(data,left,right,p)
    
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

                if d[k] >= p[k]:
                    left.append(d)
                elif d[k] < p[k]:
                    right.append(d)

            
            
            tmp_igr = information_gain(data,left,right,p)


            if tmp_igr > max_igr:
                max_igr = tmp_igr
                root_pt = p
                l=left
                r=right
                max_dim = k
    
    return root_pt,max_igr,l,r,max_dim

def iterate_tree(root,level=0):
    if level == 0:
        ret = "\t"*level+"ROOT: "+str(root.val)+f"Condition x_{root.dim} >= {root.val[root.dim]}\n" 
    else:  
        ret = "\t"*level+str(root.val)+f"Condition x_{root.dim} >= {root.val[root.dim]}\n" 
    

    if root.left.val != None:
        ret+=iterate_tree(root.left, level+1 )
    else:
        ret+="\t"*(level+1)+f"Left x_{root.dim} >= {root.val[root.dim]} LEAF:{root.left.predict()}"+"\n"
    
    if root.right.val != None:
        ret+=iterate_tree(root.right, level+1 )
    else:
        pred = root.right.predict()
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
        # print(vals,self.pool)
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

        print(f"IGR: {igr}\n LEFT:{l}\tRIGHT:{r} \t with ROOT:{root_node_0}")

        self.root = Node(root_node_0,k,data)
        self.build_tree(self.root,l,r,k)

        if name != "":
            print(f"Finished with {name}")


    def build_tree(self,root,l,r,feat):
        
        stop_l = False
        stop_r = False
        if len(r) == 0:
            r_n = Node(None,feat,[tmp_root])
            root.set_right(r_n)
            stop_r = True
        
        if len(l) == 0:
            l_n = Node(None,feat,[tmp_root])
            root.set_left(l_n)
            stop_l = True

        if not stop_l:
            tmp_root,igr,l_tmp,r_tmp,k = find_split(l)
            

            if igr == 0:
                l_n = Node(None,feat,l)
                root.set_left(l_n)
            else:
                l_n = Node(tmp_root,k,l)
                self.build_tree(l_n,l_tmp,r_tmp,k)
                root.set_left(l_n)

        

        if not stop_r:
            tmp_root,igr,l_tmp,r_tmp,k = find_split(r)

            

            if igr == 0:
                r_n = Node(None,feat,r)
                root.set_right(r_n)
            else:
                r_n = Node(tmp_root,k,r)
                self.build_tree(r_n,l_tmp,r_tmp,k)
                root.set_right(r_n)

            

    def predict(self,val,node=None):
        if node == None:
            node = self.root
            is_root = True
        else:
            is_root = False



        if val[node.dim] >= node.val[node.dim]:
            if node.left.val == None:
                return node.left.predict()
            else:
                return self.predict(val,node=node.left)
        else:
            if node.right.val == None:
                return node.right.predict()
            else:
                return self.predict(val,node=node.right)

def visualize_dataset(ds):
    
    
    labels = ["red","blue"]
    for x1,x2,y in ds:
        plt.scatter(x1,x2,c=labels[y])

    plt.title("D2 Visualization")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    
    plt.savefig("../D2_Visualization.pdf")

def big_segmentation(data):
    import random
    D_32 = []
    D_32_not = []

    D_128 = []
    D_128_not = []

    D_512 = []
    D_512_not = []

    D_2048 = []
    D_2048_not = []
    
    D_8192 = []
    D_8192_not = []

    tmp_list = []
    d_copy = data.copy()
    for i in range(0,8192+1):
        c = random.choice(data)
        ind = data.index(c)
        d_copy[ind] = -1
        if (i)%32 == 0 and i <= 32:
            D_32 = tmp_list.copy()
            tmp_list.append(c)
            D_32_not = d_copy.copy()

        elif (i)%128 == 0 and i <= 128:
            D_128 = tmp_list.copy()
            tmp_list.append(c)
            D_128_not = d_copy.copy()

        elif (i)%512 == 0 and i <= 512:
            D_512 = tmp_list.copy()
            tmp_list.append(c)
            D_512_not = d_copy.copy()

        elif (i)%2048 == 0 and i <= 2048:
            D_2048 = tmp_list.copy()
            tmp_list.append(c)
            D_2048_not = d_copy.copy()

        elif (i)%8192 == 0 and i <= 8192:
            D_8192 = tmp_list.copy()
            tmp_list.append(c)
            D_8192_not = d_copy.copy()

        else:
            tmp_list.append(c)

    print(len(D_8192))
    assert(len(D_32) == 32)
    assert(len(D_128) == 128)
    assert(len(D_512) == 512)
    assert(len(D_2048) == 2048)
    assert(len(D_8192) == 8192)
    
    return (D_32,D_32_not),(D_128,D_128_not),(D_512,D_512_not),(D_2048,D_2048_not),(D_8192,D_8192_not)
    
    

def normal_dataset_logic(dataset,key):
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
    visualize_dataset(ds)
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

def big_dataset_logic(dataset,key):
    permutations = big_segmentation(dataset[key])
    x_axis = [32,128,512,2048,8192]
    y_axis = []
    for s in permutations:
        error = 0

        training,testing = s
        t = DecisionTree(training,name=key)
        for p in testing:
            if p == -1:
                continue
        
            pred = t.predict(p)
            if pred != p[2]:
                error+=1
        loss = error/len(testing)
        y_axis.append(loss*100)
        print(f"LOSS for D_{len(training)} = {loss}")
    
    plt.plot(x_axis,y_axis)
    plt.title("Learning Curve")
    plt.xlabel("Points in Training Set")
    plt.ylabel("Percentage of Miss Classifications")
    plt.savefig("../learning_curve.pdf")

if __name__ == "__main__":
    path = "../data"
    dataset = walk_all_files(path)
    for key in dataset.keys():
        if "big" in key:
            # normal_dataset_logic(dataset,key)
            big_dataset_logic(dataset,key)