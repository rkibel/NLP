import pandas as pd
import numpy as np

def read_pandas(fname):
    df = pd.read_csv(fname)
    df = df.set_index("id")
    return df

def main():
    test_noans = read_pandas("test_noans.csv")
    train = read_pandas("train.csv")

    train_labels = train["label"].to_numpy()
    train_words = train[train.keys()[:-1]].to_numpy()

    class_vals, class_counts = np.unique(train_labels, return_counts = True)
    total_class_count = np.sum(class_counts)

    y0_words = train_words[np.argwhere(train_labels == 0).flatten()]
    y1_words = train_words[np.argwhere(train_labels == 1).flatten()]

    token_sums_0 = np.sum(y0_words, axis = 0)
    token_sums_1 = np.sum(y1_words, axis = 0)
    total_tokens_0 = np.sum(token_sums_0)
    total_tokens_1 = np.sum(token_sums_1)

    train_documents_len, train_words_len = np.shape(train_words)
    logProb = np.zeros((2, train_words_len))
    for i in range(train_words_len):
        logProb[0][i] = np.log((token_sums_0[i]+1) / (total_tokens_0 + train_words_len))
        logProb[1][i] = np.log((token_sums_1[i]+1) / (total_tokens_1 + train_words_len))

    # to test if prediction works for training labels, not necessary
    '''
    predicted_train_labels = np.zeros(train_documents_len, dtype='int')
    for i in range(train_documents_len):
        y0 = np.log(class_counts[0] / total_class_count)
        y1 = np.log(class_counts[1] / total_class_count)
        for j in range(train_words_len):
            y0 += train_words[i][j] * logProb[0][j]
            y1 += train_words[i][j] * logProb[1][j]
        if (y0 > y1): predicted_train_labels[i] = 0
        else: predicted_train_labels[i] = 1
    assert((predicted_train_labels == train_labels).all())
    '''

    test_words = test_noans.to_numpy()
    test_documents_len, test_words_len = np.shape(test_words)
    predicted_test_labels = np.zeros(test_documents_len, dtype='int')
    for i in range(test_documents_len):
        y0 = np.log(class_counts[0] / total_class_count)
        y1 = np.log(class_counts[1] / total_class_count)
        for j in range(test_words_len):
            y0 += test_words[i][j] * logProb[0][j]
            y1 += test_words[i][j] * logProb[1][j]
        if (y0 > y1): predicted_test_labels[i] = 0
        else: predicted_test_labels[i] = 1

    test_noans["label"] = predicted_test_labels
    test_ans = test_noans["label"]
    test_ans.to_csv("test_ans.csv")

if __name__ == "__main__":
    main()