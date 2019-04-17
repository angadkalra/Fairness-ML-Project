import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp

# Read dataframe
dataset = pd.read_csv('data/modified_heart_df').drop(columns='Unnamed: 0')

# Shuffle dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Split into train/test
train_df = dataset.iloc[:int(np.floor(0.7*dataset.shape[0])),:]
test_df = dataset.iloc[int(np.floor(0.7*dataset.shape[0])):, :]

# Get gender column index
sex_male_idx = list(train_df.drop(columns='target').columns).index('sex_male')

### Fairness Evaluation Functions

# Discrimination
def discrim(X_test, v, K):
    X_test_male = X_test[X_test[:,sex_male_idx] == 1]
    X_test_fem = X_test[X_test[:,sex_male_idx] == 0]
    
    y_pred_male = predict(v, X_test_male, K)
    y_pred_fem = predict(v, X_test_fem, K)
    
    discrim = y_pred_male.sum()/y_pred_male.shape[0] - y_pred_fem.sum()/y_pred_fem.shape[0]
    discrim = np.abs(discrim)
    
    return discrim

# Accuracy
def accuracy(y_true, y_pred_prob, X):
    y_pred = (y_pred_prob > 0.5)
    return (1 - 1/X.shape[0]*np.sum(np.abs(y_pred-y_true)))

# Consistency
from sklearn.neighbors import KNeighborsClassifier
def consistency(X_test, v, K):
    
    y_pred = predict(v, X_test, K) > 0.5
    
    k = 5
    knn_model = KNeighborsClassifier().fit(X_test, y_test)
    nn = knn_model.kneighbors(X_test, k, False)
    
    consist_score = 0
    for i in range(X_test.shape[0]):
        nn_sum = 0
        for j in nn[i][1:]:
            nn_sum += y_pred[j]
            
        consist_score += np.abs(y_pred[i] - nn_sum)
    
    return (1 - 1/(y_pred.shape[0]*k)*consist_score)

### LFR 

# training set
X_0 = train_df.drop(columns='target').values
y_0 = train_df.target.values

X_test = test_df.drop(columns='target').values
y_test = test_df.target.values

# Standardize train/test sets
mmscaler = MinMaxScaler().fit(X_0) 
X_0 = mmscaler.transform(X_0)
X_test = mmscaler.transform(X_test)

# validation and train sets
# X_valid = X_0[:40,:]
# y_valid = y_0[:40]

# X_0 = X_0[40:,:]
# y_0 = y_0[40:]

# training set with and without sensitive attr. == 1
X_0_pos = X_0[X_0[:,sex_male_idx] == 1]
X_0_neg = X_0[X_0[:,sex_male_idx] == 0]

N,D = X_0.shape

# Helper Functions
def dist_func(x, v, alpha):
    assert x.shape == v.shape
    assert alpha.shape == x.shape

    return np.sqrt(np.sum(((x-v)**2)*alpha))
    
# M_nk = P(Z=k|x), for all n,k. 
def softmax(x, k, alpha, Z):
    denom = 0
    for j in range(Z.shape[0]):
        denom += np.exp(-1*dist_func(x, Z[j,:], alpha))

    return np.exp(-1*dist_func(x, Z[k,:], alpha))/denom

# M_k^+
def M_pos(k, alpha, Z):
    exp_value = 0
    for i in range(X_0_pos.shape[0]):
        x = X_0_pos[i,:]
        exp_value += softmax(x, k, alpha, Z)
    
    return (1/X_0_pos.shape[0])*exp_value

# M_k^-
def M_neg(k, alpha, Z):
    exp_value = 0
    for i in range(X_0_neg.shape[0]):
        x = X_0_neg[i,:]
        exp_value += softmax(x, k, alpha, Z)
    
    return (1/X_0_neg.shape[0])*exp_value

# LFR Predict 
def predict(v, X, K):
    _,D = X.shape
    
    Z = np.reshape(v[0:K*D], (K,D))
    assert Z.shape == (K,D)
    
    w = v[K*D:K*D+K]
    assert w.shape[0] == K
    
    alpha = v[K*D+K:]
    assert alpha.shape[0] == D
    
    y_pred = []
    
    for i in range(X.shape[0]):
        x = X[i,:]
        
        y = 0
        for k in range(K):
            y += softmax(x, k, alpha, Z)*w[k]
        
        y_pred.append(y)
    
    return np.array(y_pred) 
 
# Params. and Hyper-Params  
K = 10

Z = X_0[np.random.randint(0, N, size=K),:]
Z = np.reshape(Z, (K*D,))

w = np.random.random_sample(K)
alpha = np.array([1]*D)

v_0 = np.concatenate((Z,w,alpha), axis=0)

# Loss Function
def loss_fn(v, Az, Ax, Ay):
    Z = np.reshape(v[0:(K*D)], (K,D))
    w = v[K*D:(K*D + K)]
    alpha = v[K*D + K:]
    
    # L_z
    L_z = 0
    for k in range(0,K):
        L_z += np.abs(M_pos(k, alpha, Z) - M_neg(k, alpha, Z))
        
    # L_x
    L_x = 0
    for i in range(X_0.shape[0]):
        x = X_0[i,:]
        
        x_hat = 0
        for k in range(K):
            x_hat += softmax(x, k, alpha, Z)*Z[k,:]
        
        assert x.shape == x_hat.shape
        L_x += np.sum((x - x_hat)**2) 
    
    # L_y
    L_y = 0
    for i in range(X_0.shape[0]):
        x = X_0[i,:]
        
        y_hat = 0
        for k in range(K):
            y_hat += softmax(x, k, alpha, Z)*w[k]
        
        if y_0[i]:
            L_y += -1*np.log(y_hat)
        else:
            L_y += -1*np.log(1-y_hat)
        
    
    return (Az*L_z + Ax*L_x + Ay*L_y)

# Bound w to be probabilities
w_constr = [(None,None)]*v_0.shape[0]
w_constr[K*D:(K*D + K)] = [(0,1)]*K 

from scipy.optimize import minimize

results = []
def min_lossfunc(az,ax,ay):
    v = minimize(loss_fn, v_0, args=(az,ax,ay), method='L-BFGS-B', 
                options={'maxiter': 30, 'disp': True}, bounds=w_constr).x

    y_pred_prob = predict(v, X_valid, K)
    acc = accuracy(y_valid, y_pred_prob, X_valid)
    discr = discrim(X_valid, v, K)
    max_delta = acc - discr
    consis = consistency(X_valid, v, K)

    return [az, ax, ay, acc, discr, max_delta, consis]

def collect_result(res):
    global results
    results.append(res)

# All HP combinations
from itertools import permutations
perm = list(permutations([0.1, 0.5, 1, 5, 10], 3))

def train_lfr(pool):
    for p in perm:
        pool.apply_async(min_lossfunc, args=(p), callback=collect_result)
    # pool.apply_async(min_lossfunc, args=((1,1,1)), callback=collect_result)
    # pool.apply_async(min_lossfunc, args=((0.1,1,10)), callback=collect_result)
    # pool.apply_async(min_lossfunc, args=((10,1,0.1)), callback=collect_result)
    # pool.apply_async(min_lossfunc, args=((0.1,10,1)), callback=collect_result)
    # pool.apply_async(min_lossfunc, args=((1,10,0.1)), callback=collect_result)
    pool.close()
    pool.join()

    results = np.array(results)
    np.savetxt('train_results_all', results, fmt='%2.2f, %2.2f, %2.2f, %0.3f, %0.3f, %0.3f, %0.3f', newline='\n')

def lfr_predict(opt_hp):
    # retrain on full training set, predict on test set, output metrics
    v_opt = minimize(loss_fn, v_0, args=opt_hp, method='L-BFGS-B', 
                options={'maxiter': 30, 'disp': True}, bounds=w_constr).x

    y_pred_prob = predict(v, X_test, K)
    acc = accuracy(y_test, y_pred_prob, X_test)
    discr = discrim(X_test, v, K)
    max_delta = acc - discr
    consis = consistency(X_test, v, K)

    test_res = np.array([acc, discr, max_delta, consis])
    np.savetxt("test_results_all", test_res, fmt='%0.3f, %0.3f, %0.3f, %0.3f', newline='\n')

if __name__ == '__main__':
    # pool = mp.Pool(mp.cpu_count())
    # train_lfr(pool)
    lfr_predict((10, 1, 0.1))
    

    
