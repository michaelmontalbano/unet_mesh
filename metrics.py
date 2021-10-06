from numpy.core.fromnumeric import mean
from display import *
from tensorflow import keras
import visualkeras
import shap
import seaborn as sns
import pandas as pd
from skimage.metrics import structural_similarity as simm
from sklearn.metrics import mean_absolute_error

# Load in pickle
#f= 'filters_[16, 32, 64]_guass_0.1_unet_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_raw_opt_adam_regression_thres_0.25'
fOld = 'standardScalar_guass_0.1_unet_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_raw_opt_adam_regression'
f_mlse = 'MLSE_ins_raw_epochs_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_opt_adam_regression_thres_0.25'
f = 'old_ins_raw_noNSE_epochs_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_opt_adam_regression_thres_0.25'

# Scale correctly
r_noNSE = open_pickle('results/{}_results.pkl'.format(f))
r = open_pickle('results/{}_results.pkl'.format(fOld))
r_mlse = open_pickle('results/{}_results.pkl'.format(f_mlse))

y_true = r['true_testing']
y_pred = r['predict_testing']

#mse, indices = get_mse(r,'testing')
#differences, maxs, predmaxs = check_accuracy(r)

# ***************************************************
# Experiments with SIMM 
# not too revealing with the continous data

#good = [idx for idx, element in enumerate(predmaxs) if abs(maxs[idx] - element) < 10 and maxs[idx] > 20]

def get_simm_list(r):
    # structural similarity metric
    simms = []
    for idx1, idx2 in enumerate(np.arange(0,len(y_true))):
        img_true = np.squeeze(r['true_testing'][int(idx2)])
        img_pred = np.squeeze(r['predict_testing'][int(idx2)])
        simm_ = simm(img_true, img_pred, data_range=img_pred.max()-img_pred.min())
        simms.append([simm_,idx1])
    simms = np.asarray(simms)
    simms = sorted(simms,key=lambda x: x[0])
    return simms

# badlow = [idx for idx, element in enumerate(predmaxs) if element < 15 and maxs[idx] > 20 and maxs[idx] < 30]

# simms = []
# for idx in badlow:
#     img_true = np.squeeze(r['true_testing'][int(idx)])
#     img_pred = np.squeeze(r['predict_testing'][int(idx)])
#     simm_ = simm(img_true, img_pred, data_range=img_pred.max()-img_pred.min())
#     simms.append(simm_)

# print(simms[:10])
# print(simms[-50:-40])
# simms = get_simm_list(r)
plt.figure(1)
#plt.hist([x for x, _ in simms])
plt.savefig('no_nse.png')
plt.show()

# ***************************************************
# explained variance
import sklearn.metrics as sk
'''
trueFlat = y_true.flatten()
predFlat = y_pred.flatten()
oldFlat = y_predOld.flatten()
trueNewFlat = y_trueNew.flatten()
newFlat = y_predNew.flatten()
sk.explained_variance_score(trueFlat,oldFlat)
'''
# ***************************************************
# contains methods to score u-nets

def count_all(pictures):
    '''
    make a list of all pixels
    input: np array of shape (x, 1, 60, 60)
    example usage: count_all(ins[:,17,:,:])
    '''
    pixels = []
    for ex in pictures:
        for row in ex:
            for pixel in row:
                pixels.append(pixel)
    return pixels

# F1 Score
#sklearn.metrics.f1_score(y_true,y_pred)

# bias = 1/n sum(f_i) / i/n sum(o_i)

# MAE, RMSE, CC, S1 score 

# we could make our own definition for what is considered a hit

# critical success index
# = hits/total number of forecasts

# false alarm ratio (FAR)
def stats(r, scaler):
    correct, total, TP, FP, TN, FN, events = binary_accuracy(r,scaler)
    FAR = FP/(FP + TN)
    # probability of detection
    POD = TP/(events)
    mse, indices = get_mse(r)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    #mae = mean_absolute_error(y_true,y_pred)
    #max_error = sk.max_error(y_true,y_pred)
    #mse = sk.mean_absolute_error(y_true,y_pred)
    #msle = sk.mean_squared_log_error(y_true,y_pred)
    #r2 = sk.r2_score(y_true,y_pred)


    return FAR, POD, accuracy #, mae, max_error, mse, msle, r2

#FAR_old, POD_old, acc_Old, mae_Old, mse_Old, max_er_Old, mse_Old, msle_Old, r2 = stats(r_noNSE)
#print("FAR for original model",FAR_old, POD_old, acc_Old, mae_Old, mse_Old, max_er_Old, mse_Old, msle_Old, r2)
#print("FAR for model w/o NSE",stats(r))

# probability of detection

# Equitable Threat Score

# Accuracy

# Structural Similarity


