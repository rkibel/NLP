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
    train_documents_len, train_words_len = np.shape(train_words)
    
    test_words = test_noans.to_numpy()
    test_documents_len, test_words_len = np.shape(test_words)

    train_labels[np.argwhere(train_labels == 0)] = -1

    T = 10
    t = 0
    k = 1
    c = np.zeros(2)
    w = np.zeros((2, train_words_len))

    while t <= T:
        for i in range(train_documents_len):
            if (train_labels[i] * np.dot(w[k], train_words[i]) <= 0):
                w = np.vstack([w, np.array(w[k] + train_labels[i] * train_words[i])])
                c = np.append(c, 1)
                k += 1
            else:
                c[k] += 1
        t += 1

    # to test if prediction works for training labels, not necessary
    '''
    predicted_train_labels = np.zeros(train_documents_len, dtype='int')
    for i in range(train_documents_len):
        s = 0
        for j in range(1, k+1):
            s += c[j]*np.sign(np.dot(train_words[i], w[j]))
        predicted_train_labels[i] = np.sign(s)
    '''

    predicted_test_labels = np.zeros(test_documents_len, dtype='int')
    for i in range(test_documents_len):
        s = 0
        for j in range(1, k+1):
            s += c[j]*np.sign(np.dot(test_words[i], w[j]))
        predicted_test_labels[i] = np.sign(s)

    predicted_test_labels[np.argwhere(predicted_test_labels == -1)] = 0

    test_noans["label"] = predicted_test_labels
    test_ans = test_noans["label"]
    test_ans.to_csv("test_ans.csv")
    
if __name__ == "__main__":
    main()