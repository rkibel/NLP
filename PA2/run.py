import pandas as pd
import numpy as np
from numpy.linalg import norm

def read_pandas(fname):
    df = pd.read_csv(fname)
    df = df.set_index("id")
    return df

def train_model(train_tot, alpha, lambd, epsilon, alpha_decay):
    w = np.zeros(np.shape(train_tot)[1] - 1)
    p = 0.0
    LCL = 1.0
    for epoch in range(1000):
        oldval = LCL - lambd * norm(w, 2)
        LCL = 0.0
        np.random.shuffle(train_tot)
        for example in train_tot:
            x = example[:-1]
            y = example[-1]
            p = 1.0 / (1.0 + np.exp(-np.dot(w, x)))
            w = w + alpha * ((y-p) * x - 2 * lambd * w)
            LCL += y * np.log(p) + (1-y) * np.log(1-p)
        alpha *= alpha_decay
        newval = LCL - lambd * norm(w, 2)
        if (abs(newval - oldval) < epsilon): break
        print(newval)
    
    return w

def test_model(w, test_features, test_labels = [None]):
    test_id_len = np.shape(test_features)[0]
    predicted_test_labels = np.zeros(test_id_len, dtype='int')

    for i in range(test_id_len):
        x = test_features[i]
        p1 = 1.0 / (1.0 + np.exp(-np.dot(w, x)))
        p0 = 1.0 - p1
        if p0 > p1: predicted_test_labels[i] = 0
        else: predicted_test_labels[i] = 1
            
            
    acc = 0.0
    if test_labels[0] != None:
        diff = predicted_test_labels - test_labels
        acc = 1.0 - np.dot(diff, diff) / test_id_len
        
        print("predicted labels:", predicted_test_labels)
        print("true labels:", test_labels)
        print("accuracy:", acc)
        
    return predicted_test_labels, acc

def fold10cv(train_tot, alpha, lambd, epsilon, alpha_decay):
    train_id_len = np.shape(train_tot)[0]
    accuracies = list()
    for i in range(10):
        train_segment = np.concatenate([np.arange(0, train_id_len*i/10.0, dtype = "int"), np.arange(train_id_len*(i+1)/10.0, train_id_len, dtype = "int")])
        test_segment = np.arange(train_id_len*i/10.0, train_id_len*(i+1)/10.0, dtype = "int")
        
        train_total = np.copy(train_tot[train_segment])
        w = train_model(train_total, alpha, lambd, epsilon, alpha_decay)
        test_total = np.copy(train_tot[test_segment])
        predicted_test_labels, a = test_model(w, test_features = test_total[:, :-1], test_labels = test_total[:, -1])
        print(a)
        accuracies.append(a)
        
    return accuracies

def main():
    test_noans = read_pandas("test_noans.csv")
    train = read_pandas("train.csv")
    train_tot = train[train.keys()].to_numpy()
    test_features = test_noans.to_numpy()
    
    w = train_model(train_tot, alpha = 0.0495, lambd = 1e-5, epsilon = 1e-1, alpha_decay = 0.975)
    
    # to test if prediction works for training labels, not necessary
    '''
    predicted_test_labels, acc = test_model(w, test_features = train_tot[:, :-1], test_labels = train_tot[:, -1])
    print(acc)
    '''
    
    predicted_test_labels, acc = test_model(w, test_features = test_features)
    test_noans["label"] = predicted_test_labels
    test_ans = test_noans["label"]
    test_ans.to_csv("test_ans.csv")
    
if __name__ == "__main__":
    main()

