import numpy as np


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
    max_igr = -1
    root_pt = (0,0,-1)
    l=[]
    r=[]
    max_dim = 0

    if force_node:
        p = data[force_ind]
        k = 0 
        left = []
        right = []

        for j,d in enumerate(data):

            if d[k] >= p[k]:
                left.append(d)
            elif d[k] < p[k]:
                right.append(d)
                
                
        print(left,right,k) 
        tmp_igr = information_gain(data,left,right,p)

        if tmp_igr > max_igr:
            max_igr = tmp_igr
            root_pt = p
            l=left
            r=right
            feat = k

        root_pt = p
        max_igr = information_gain(data,left,right,p)
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
