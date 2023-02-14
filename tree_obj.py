from utils import *
from node import *

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

class DecisionTree():
    def __init__(self,data,name="",force_choice=False,choice_ind=0):
        self.data = data
        self.node_count = 1

        if force_choice:
            root_node_0,igr,l,r,k =  find_split(data,force_node=True,force_ind=choice_ind)
        else:
            root_node_0,igr,l,r,k =  find_split(data)

        self.root = Node(root_node_0,k,data)
        self.build_tree(self.root,l,r,k)

        if name != "":
            print(f"Finished with {name}")


    def build_tree(self,root,l,r,feat):
        
        stop_l = False
        stop_r = False
        if len(r) == 0:
            r_n = Node(None,feat,[root.val])
            root.set_right(r_n)
            stop_r = True
            self.node_count+=1
        
        if len(l) == 0:
            l_n = Node(None,feat,[root.val])
            root.set_left(l_n)
            stop_l = True
            self.node_count+=1

        if not stop_l:
            tmp_root,igr,l_tmp,r_tmp,k = find_split(l)
            
            # print(f"IGR: {igr}\n LEFT:{l_tmp}\tRIGHT:{r_tmp} \t with ROOT:{tmp_root}\tDIM:{k}")
            if igr == 0:
                l_n = Node(None,feat,l)
                self.node_count+=1
                root.set_left(l_n)
            else:
                l_n = Node(tmp_root,k,l)
                self.node_count+=1
                self.build_tree(l_n,l_tmp,r_tmp,k)
                root.set_left(l_n)

        

        if not stop_r:
            tmp_root,igr,l_tmp,r_tmp,k = find_split(r)

            

            if igr == 0:
                # print(r,l,root.val,tmp_root)
                r_n = Node(None,feat,r)
                self.node_count+=1
                root.set_right(r_n)
            else:
                # print(r,l,root.val,tmp_root)
                r_n = Node(tmp_root,k,r)
                self.build_tree(r_n,l_tmp,r_tmp,k)
                root.set_right(r_n)
                self.node_count+=1

            

    def predict(self,val,node=None,level=0):
        if node == None:
            node = self.root
            is_root = True
        else:
            is_root = False
        
        
        # node.get_condition()
        
        if node.val == None:
            # print(f"POOL: {node.pool}\tVAL:{val}\tLevel:{level}")
            return node.predict()

        # print(f"POOL: {node.pool}\tRIGHT POOL: {node.right.pool}\tLEFT POOL: {node.left.pool}\tVAL:{val}\tLevel:{level}")
        
        if val[node.dim] >= node.val[node.dim]:
            # print(f"LEFT")
            if node.val == None:
                return node.predict()
            else:
                return self.predict(val,node=node.left,level=level+1)
        else:
            # print(f"RIGHT")
            if node.val == None:
                return node.predict()
            else:
                return self.predict(val,node=node.right,level=level+1)
