import pandas as pd
import numpy as np



def ez_pandas_reader(fname):
    df = pd.read_csv(fname)
    df = df.set_index("id")
    return df


def main():
    # main code goes here
    test_noans = ez_pandas_reader("test_noans.csv")
    train = ez_pandas_reader("train.csv")

    train_labels = train["label"].to_numpy()
    vals, counts = np.unique(train_labels, return_counts = True)

    max_count_val = np.argwhere(counts == np.max(counts)).flatten()[0]
    print(max_count_val)
    output_labels = np.array(100 * [max_count_val])
    
    print(type(test_noans))
    test_noans["label"] = output_labels


    print(type(test_noans))

    test_ans = test_noans["label"]
    test_ans.to_csv("dumb_output_2.csv")

    print("it worked!!!!!!!!")



if __name__ == "__main__":
    main()
