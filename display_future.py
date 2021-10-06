# michael montalbano
# 8/24/2021

# functions for display
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from sklearn.model_selection import train_test_split
import re
import pickle
from ast import literal_eval
import sklearn.metrics
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import figure
import matplotlib.cm as cm
from matplotlib import colors

DATA_HOME='/mnt/data/SHAVE_cases'
OUT_HOME='/mnt/data/michaelm/practicum/cases'
comp_max_file = "{}/{}/multi{}/MergedReflectivityQCComposite_Max_30min/01.00/20150806-{}00.netcdf"
uncropped_mesh_file = '{}/{}/multi{}/uncropped/Reflectivity_-20C_Max_30min/01.00/{}-{}00.netcdf'
cropped_mesh_file = '{}/{}/multi{}/storm{}/MESH_Max_30min/01.00/{}-{}00.netcdf'
target_file = '{}/{}/multi{}/storm{}/target_MESH_Max_30min/MESH_Max_30min/01.00/{}-{}00.netcdf'

NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km']
multi_fields = ['MergedReflectivityQCComposite_Max_30min','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min']
products = NSE_fields + multi_fields

ins = np.load('ins_raw.npy')

def open_pickle(file):
    fp = open(file,'rb')
    r = pickle.load(fp)
    fp.close()
    return r

scalers = open_pickle('scaler.pkl')
scaler = scalers[0]

def plot_predict(r,idx, group='testing'):
    f, axs = plt.subplots(1,2,figsize=(15,15))
    
    plt.subplot(121)
    true = r['true_{}'.format(group)][idx]
    slc = true.reshape(1, 60*60)
    transformed = scaler.inverse_transform(slc)
    transformed = transformed.reshape(60, 60, 1) # reshape it back to tiles
    ax = plt.gca()
#    im = ax.imshow(transformed)
    im = ax.imshow(transformed,cmap=cm.nipy_spectral)
    plt.xlabel('Degrees Longitude')
    plt.ylabel('Degrees Latitude')
    plt.title('True MESH {} Set {}'.format(group,idx))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(122)
    pred = r['predict_{}'.format(group)][idx]
    slc = pred.reshape(1, 60*60)
    transformed = scaler.inverse_transform(slc)
    transformed = transformed.reshape(60, 60, 1) # reshape it back to tiles
    ax = plt.gca()
    im = ax.imshow(transformed,cmap=cm.nipy_spectral)
    plt.title('Predicted MESH {} Set {}'.format(group,idx))
    plt.xlabel('Degrees Longitude')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def plot_extra(r,idx): 
    # add more plots
    # make colorbars consistent
    # cmap = mpl.cm.cool
    # norm = mpl.colors.Normalize(vmin=0,vmax=100)
    # cbl = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='vertical')



    idx = r['outs_test_indices'][idx]
    f, axs = plt.subplots(2,3,figsize=(15,15))

    plt.subplot(321)
    ax = plt.gca()
    bounds = [0.000,0.003,0.006,0.009,0.012]
    h = ins[idx][13]
    cs = plt.contourf(h, levels=[0.000,0.003,0.006,0.009,0.012], colors=["#41ead6", "#1a2ee8", "#28ed32", "#eda828", "#ff2929"], extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.title('LL Shear Max Swath')


    plt.subplot(322)
    ax = plt.gca()
    bounds = [0.000,0.003,0.006,0.009,0.012]
    h = ins[idx][14]
    cs = plt.contourf(h, levels=[0.000,0.003,0.006,0.009,0.012], colors=["#41ead6", "#1a2ee8", "#28ed32", "#eda828", "#ff2929"], extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.title('ML Shear Max Swath')   

    plt.subplot(323)
    ax = plt.gca()
    bounds = [0,15,30,45,60,75]
    h = ins[idx][17]
    cs = plt.contourf(h, levels=[0,15,30,45,60,75], colors=["#41ead6", "#1a2ee8", "#28ed32", "#eda828", "#ff2929"], extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.title('MESH Swath 30 min prior')
    
    plt.subplot(324)
    ax = plt.gca()
    bounds = [0,15,30,45,60,75]
    h = ins[idx][12]
    cs = plt.contourf(h, levels=[0,15,30,45,60,75], colors=["#41ead6", "#1a2ee8", "#28ed32", "#eda828", "#ff2929"], extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.title('Composite Reflectivity')
    
    plt.subplot(325)
    ax = plt.gca()
    bounds = [0,15,30,45,60,75]
    h = ins[idx][0]
    cs = plt.contourf(h, levels=[0,15,30,45,60,75], colors=["#41ead6", "#1a2ee8", "#28ed32", "#eda828", "#ff2929"], extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.title('0-6km shear')
    
    plt.subplot(326)
    ax = plt.gca()
    bounds = [0,15,30,45,60,75]
    h = ins[idx][22]
    cs = plt.contourf(h, levels=[0,15,30,45,60,75], colors=["#41ead6", "#1a2ee8", "#28ed32", "#eda828", "#ff2929"], extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.title('2 Deg Ref')

    # wind profile? 

def plot(file,field="MESH_Max_30min",xlim=None,ylim=None):
    
    print(file)
    f = Dataset(file,mode='r')
    var = f.variables[field][:,:] # read in field
    var = np.where(var<-1000,0,var)
    plt.imshow(var)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.colorbar()

def get_mse(r,group='testing'):

    mse_list = []
    indices = np.arange(0,len(r['true_{}'.format(group)]),1)
    mse = 10000
    for idx, example in enumerate(indices):
        y_true = r['true_{}'.format(group)][idx]
        y_pred = r['predict_{}'.format(group)][idx]
        y_true = np.where(y_true<0.1,np.nan,y_true)
        y_pred = np.where(y_true<0.1,np.nan,y_pred)

        y_true = np.ma.array(y_true,mask=np.isnan(y_true),fill_value=10)
        y_pred = np.ma.array(y_pred,mask=np.isnan(y_pred),fill_value=10)

        difference_array = np.subtract(y_true,y_pred)
        squared_array = np.square(difference_array)
        mse = squared_array.mean()
        mse_list.append([mse,idx])
    
    indices = []
    mse_list_2 = []
    for ex in mse_list:
        if np.ma.is_masked(ex[0]):
            mse_list_2.append([1000,ex[1]])
        else:
            mse_list_2.append([ex[0],ex[1]])
            
    mse_list_2 = np.asarray(mse_list_2)
    mse_list_2 = sorted(mse_list_2,key=lambda x: x[0]) # sort

    mse = []
    indices = []
    for ex in mse_list_2:
        mse.append(ex[0])
        indices.append(ex[1])
    return mse, indices

def check_accuracy(r,group='testing'):
    y_true = r['true_testing']
    y_pred = r['predict_testing']
    maxs = []
    pred_maxs = []

    differences = []
    for idx, ex in enumerate(y_true):
        ex = ex.reshape(1, 60*60)
        ex_pred = y_pred[idx].reshape(1,60*60)
        ex = scaler.inverse_transform(ex)
        ex_pred = scaler.inverse_transform(ex_pred)

        max_MESH = ex.max()
        pred_MESH = ex_pred.max()
        diff = abs(max_MESH - pred_MESH)
        differences.append(diff)
        maxs.append(max_MESH)
        pred_maxs.append(pred_MESH)
    return differences, maxs, pred_maxs

def binary_accuracy(r,group='testing'):
    y_true = r['true_testing']
    y_pred = r['predict_testing']

    # scale data back to mm
    correct = 0
    negative = 0
    total = 0
    positive = 0
    for idx, ex in enumerate(y_true):
        ex = ex.reshape(1, 60*60)
        ex_pred = y_pred[idx].reshape(1,60*60)
        ex = scaler.inverse_transform(ex)
        ex_pred = scaler.inverse_transform(ex_pred)
        
        # binarize the data
        ex_pred = np.where(ex_pred < 19, 0, 1) 
        ex = np.where(ex < 19, 0, 1)

        for idx, row in enumerate(ex):
            for idx2, pixel in enumerate(row):
                pred_pix = ex_pred[idx][idx2]
                if pixel == pred_pix:
                    correct+=1
                    if pixel == 1:
                        positive+=1
                if pixel != pred_pix and pixel == 1:
                    negative +=1
                total+=1
    return correct, total, positive, negative

def make_df(r,indices):
    df = pd.DataFrame(columns=products)

    for i, idx in enumerate(indices): 
        maxes = []
        for ex in ins[int(idx)]:
            ex[ex == 0] = np.nan
            maxes.append(np.nanmean(ex))
        columns = {}
        for i in range(len(df.columns)):
            columns[df.columns[i]] = maxes[i]
        df = df.append(columns, ignore_index = True)

    means=[]
    for column in df.columns:
        means.append(df[column].mean())
    columns = {}
    for i in range(len(df.columns)):
        columns[df.columns[i]] = means[i]
    df = df.append(columns, ignore_index = True)    

    return df