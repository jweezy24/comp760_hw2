import os

def walk_all_files(root_path):
    loaded_data = {}
    for root,dirs,files in os.walk(root_path):
        if "__MACOSX" in root:
            continue
        for file in files:
            if ".txt" in file:
                loaded_data[f"{root}/{file}"] = parse_data(f"{root}/{file}")
    return loaded_data

def parse_data(path):
    space = []
    with open(path,"r+") as f:
        for line in f.readlines():
            x1,x2,y = line.split(" ")
            x1 = float(x1)
            x2 = float(x2)
            y = int(y.strip())
            space.append((x1,x2,y))
    return space

if __name__ == "__main__":
    path = "data"
    dataset = walk_all_files(path)
    print(dataset)