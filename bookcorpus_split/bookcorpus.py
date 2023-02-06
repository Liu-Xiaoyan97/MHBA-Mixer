import datasets as datasets
import json
if __name__ == "__main__":
    bookcorpus_data = datasets.load_dataset("bookcorpus", split="train")
    subset = []
    for i, data in enumerate(bookcorpus_data):
        subset.append(data["text"]+"\n")
        print(i, data["text"])
        if i % 1e5 == 0:
            with open("E:\\bookcorpus_subsets\\bookcorpus_{}.txt".format(i//1e6), "w") as f:
                f.writelines(subset)
            subset = []
            print("bookcorpus_{}.txt have been written".format(i))