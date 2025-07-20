import pickle

if __name__ == "__main__":
    # label_list = "/home/jc/Documents/Data/GTAV/gtav_label_info.p.new"
    label_list = "datasets/gtav/gtav_label_info.p"
    with open(label_list, "rb") as f:
        x = pickle.load(f)
        print(len(x[1]))
        for i in x[0]:
            print(len(i))
    
        print("load file")