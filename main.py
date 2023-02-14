import numpy as np
from data_parser import *
from tree_obj import *
import matplotlib.pyplot as plt



def visualize_dataset(ds):
    
    
    labels = ["red","blue"]
    for x1,x2,y in ds:
        plt.scatter(x1,x2,c=labels[y])

    plt.title("Bad Dataset Visualization")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    
    plt.savefig("../Question2.pdf")

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
    
def visualize_boundaries(t,data,plot_title=""):
    unzipped = ([ a for a,b,c in data ], [ b for a,b,c in data], [ c for a,b,c in data] )
    f1 = unzipped[0]
    f2 = unzipped[1]
    min1 = min(f1)
    max1 = max(f1)
    min2 = min(f2)
    max2 = max(f2)
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1,r2))
    zz = []
    print(grid.shape)
    for x,y in grid:
        c = t.predict((x,y))
        zz.append(c)
    zz =np.array(zz)
    zz = zz.reshape(xx.shape)
    print(zz.shape,xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')
    plt.title(f"D_{plot_title} Boundry")   
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1") 
    plt.savefig(f"../{plot_title}.png")

def normal_dataset_logic(dataset,key):
    t = DecisionTree(dataset[key],name=key,force_choice=False,choice_ind=2)
    print(iterate_tree(t.root))
    ds = dataset[key]
    error = 0
    X = []
    Y = []
    for d in ds:
        pred = t.predict(d)
        if pred != d[2]:
            error+=1
            # print(f"ERROR AT {d}")
    loss = error/len(ds)
    print(f"LOSS = {loss}")
    visualize_dataset(ds)
    # visualize_boundaries(t,ds)

def sklearn_tree_analysis(dataset,key):
    from sklearn import tree
    permutations = big_segmentation(dataset[key])
    x_axis = [32,128,512,2048,8192]
    y_axis = []
    i=0
    for s in permutations:
        X = []
        Y = []
        training,testing = s
        for d in training:
            X.append((d[0],d[1]))
            Y.append(d[2])

    
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X,Y)
        error = 0
        for d in testing:
            if d == -1:
                continue
            x = (d[0],d[1])
            p = clf.predict([x])
            if p[0] != d[2]:
                error+=1
        loss = error/len(testing)
        y_axis.append(loss*100)
        print(f"{x_axis[i]} = {clf.tree_.node_count}")
        i+=1
    plt.plot(x_axis,y_axis)
    plt.title("Learning Curve Sklearn")
    plt.xlabel("Points in Training Set")
    plt.ylabel("Percentage of Miss Classifications")
    plt.savefig("../learning_curve_sklearn.pdf")

def big_dataset_logic(dataset,key):
    permutations = big_segmentation(dataset[key])
    x_axis = [32,128,512,2048,8192]
    y_axis = []
    i = 0
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
        visualize_boundaries(t,dataset[key],f"{x_axis[i]}")
        print(f"LOSS for D_{len(training)} = {loss}")
        print(f"{x_axis[i]} = {t.node_count}")
        i+=1
    plt.plot(x_axis,y_axis)
    plt.title("Learning Curve")
    plt.xlabel("Points in Training Set")
    plt.ylabel("Percentage of Miss Classifications")
    plt.savefig("../learning_curve.pdf")

if __name__ == "__main__":
    path = "./data"
    dataset = walk_all_files(path)
    for key in dataset.keys():
        if "bad" in key:
            normal_dataset_logic(dataset,key)
            # big_dataset_logic(dataset,key)
            # sklearn_tree_analysis(dataset,key)