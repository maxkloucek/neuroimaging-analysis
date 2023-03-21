import numpy as np
from tqdm import tqdm

def firth_logit(X,y,num_iter=5000,learning_rate=0.01):

    intercept_data = np.ones(X.shape[0], dtype=int)
    # X[:,:-1] = intercept_data
    # the intercept thing clearly dones't work...
    # X = np.c_[X, intercept_data]

    weights = np.ones(X.shape[1])
    # weights = np.zeros(X.shape[1])
    ws = np.zeros((num_iter, weights.shape[0]))

    #Define get_predictions function
    def get_predictions(X,weights):
        z = np.dot(X,weights)
        y_pred =  1/(1 + np.exp(-z))
        return y_pred
    #Perform gradient descent
    for i in tqdm(range(num_iter)):
        
        y_pred = get_predictions(X,weights)
        # print(y_pred.shape)
        #Calculate Fisher information matrix
        Xt = X.transpose()
        W = np.diag(y_pred*(1-y_pred))
        I = np.linalg.multi_dot([Xt,W,X])
        
        #Find diagonal of Hat Matrix
        sqrtW = W**0.5
        H = np.linalg.multi_dot([sqrtW,X,np.linalg.inv(I),Xt,sqrtW])
        hat_diag = np.diag(H)
        
        #Calculate U_star
        U_star = np.matmul((y -y_pred + hat_diag*(0.5 - y_pred)),X)
        
        #Update weights
        # print(weights)
        weights += np.matmul(np.linalg.inv(I),U_star)*learning_rate
        ws[i, :] = weights
    
    #Get final predictions
    # y_pred =  get_predictions(X,weights)
    return ws