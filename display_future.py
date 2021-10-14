# michael montalbano
# 8/24/2021

# functions for display
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from ast import literal_eval
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import figure
import matplotlib.cm as cm
from matplotlib import colors
import sys
from matplotlib.colors import rgb2hex

DATA_HOME='/mnt/data/SHAVE_cases'
OUT_HOME='/mnt/data/michaelm/practicum/cases'
comp_max_file = "{}/{}/multi{}/MergedReflectivityQCComposite_Max_30min/01.00/20150806-{}00.netcdf"
uncropped_mesh_file = '{}/{}/multi{}/uncropped/Reflectivity_-20C_Max_30min/01.00/{}-{}00.netcdf'
cropped_mesh_file = '{}/{}/multi{}/storm{}/MESH_Max_30min/01.00/{}-{}00.netcdf'
target_file = '{}/{}/multi{}/storm{}/target_MESH_Max_30min/MESH_Max_30min/01.00/{}-{}00.netcdf'

NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km']
multi_fields = ['MergedReflectivityQCComposite_Max_30min','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min']
products = NSE_fields + multi_fields

shear_colors = ['#202020','#808080','#4d4d00','#636300','#bbbb00','#dddd00','#ffff00','#770000','#990000','#bb0000','#dd0000','#ff0000','#ffcccc']
shear_bounds = [0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]

MESH_colors = ['#aaaaaa','#00ffff','#0080ff','#0000ff','#007f00','#00bf00','#00ff00','#ffff00','#bfbf00','#ff9900','#ff0000','#bf0000','#7f0000','#ff1fff']
MESH_bounds = [9.525,15.875,22.225,28.575,34.925,41.275,47.625,53.975,60.325,65,70,75,80,85]

uwind_r = ['0x00','0x00','0x00','0x00','0x00','0xbf','0xff','0xff','0xff','0xbf','0x7f','0xff']
uwind_g = ['0x80','0x00','0x7f','0xbf','0xff','0xff','0xff','0xbf','0x99','0x00','0x00','0x33']
uwind_b = ['0xff','0xff','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0x00','0xff']

ref_r = [0,115,120,148,2,17,199,184,199,199,153,196,122,199]
ref_g = [0,98,120,164,199,121,196,143,113,0,0,0,69,199]
ref_b = [0,130,120,148,2,1,2,0,0,0,16,199,161,199]

ref_levels = [-10,10,13,18,28,33,38,43,48,53,63,68,73,77]
wind_levels = [-30,-25,-20,-15,-10,-5,-1,1,5,10,15,20,25,30]

wind_colors = []
ref_colors = []
for idx, color in enumerate(uwind_r):
    wind_colors.append(str('#%02x%02x%02x' % (int(color,16),int(uwind_g[idx],16),int(uwind_b[idx],16))))
    ref_colors.append(str('#%02x%02x%02x' % (ref_r[idx],ref_g[idx],ref_b[idx])))

ins = np.load('ins_filtered_noRef.npy')
# no reflectivity, no Heightof-50C

def open_pickle(file):
    fp = open(file,'rb')
    r = pickle.load(fp)
    fp.close()
    return r



scalers = open_pickle('scaler_raw_noShear.pkl')
scaler = scalers[0]

def plot_predict(r,idx, scaler,group='testing'):
    f, axs = plt.subplots(1,2,figsize=(15,15))
    
    scalers = open_pickle('scaler_raw.pkl')
    scaler = scalers[0]
    plt.subplot(121)
    true = r['true_{}'.format(group)][idx]
    print(true.mean())
    slc = true.reshape(1, 60*60)
    transformed = scaler.inverse_transform(slc)
    transformed = transformed.reshape(60, 60, 1) # reshape it back to tiles
    print(transformed.mean())
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
    plt.savefig('001.png')

f = 'binary_test_ins_raw_epochs_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_opt_adam_regression_thres_None_results.pkl'
r = open_pickle('results/{}'.format(f))




def plot_extra(r,idx,other_idx, group='testing'): 
    # add more plots
    # make colorbars consistent
    # cmap = mpl.cm.cool
    # norm = mpl.colors.Normalize(vmin=0,vmax=100)
    # cbl = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='vertical')
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5),  constrained_layout=True,subplot_kw=dict(xticks=[], yticks=[], aspect=1))
    mse, indices = get_mse(r,'testing')
    f, axs = plt.subplots(2,4,figsize=(10,14))

    plt.subplot(421)
    ax = plt.gca()
    bounds = MESH_bounds
    true = r['true_{}'.format(group)][idx]
    slc = true.reshape(1, 60*60)
    transformed = scaler.inverse_transform(slc)
    transformed = transformed.reshape(60, 60, 1) # reshape it back to tiles
    x = np.squeeze(transformed)
    cs = plt.contourf(x, levels=bounds, colors=MESH_colors, extend='both',orientation='horizontal', shrink=0.5, spacing='proportional')   

    plt.colorbar(cs, ticks=bounds)
    f.tight_layout(pad=3.0)
    plt.ylabel('y (1/2 km)')
    plt.xlim([0,60])
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])
    plt.title('True MESH {} Set #{} (mm)'.format(group,idx))


########################################################################
    plt.subplot(422)
    pred = r['predict_{}'.format(group)][idx]
    slc = pred.reshape(1, 60*60)
    transformed = scaler.inverse_transform(slc)
    transformed = transformed.reshape(60, 60, 1) # reshape it back to tiles

    x = np.squeeze(transformed)
    bounds = MESH_bounds
    cs = plt.contourf(x, levels=bounds, colors=MESH_colors, extend='both')    
    plt.colorbar(cs, ticks=bounds)
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])
    plt.title('Predicted MESH {} Set #{} (mm)'.format(group,idx))
    plt.ylabel('y (1/2 km)')

    idx=other_idx
    plt.subplot(423)
    ax = plt.gca()
    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds=[0,50,100]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    # img = ax.imshow(ins[idx][13].reshape(60,60,1),interpolation='nearest', origin='lower',
    #                 cmap=cmap, norm=norm)
    x = ins[idx][1].reshape(60,60,1)
    x = np.squeeze(x)
    cs = ax.contourf(x, levels=shear_bounds,colors=shear_colors,extend='both')
    plt.title('LL Shear Max Swath (1/s)')
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])

    plt.ylabel('y (1/2 km)')
    #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 50, 100])
    plt.colorbar(cs, ticks=shear_bounds)

    
    plt.subplot(424)
    ax = plt.gca()
    x = ins[idx][2].reshape(60,60,1)
    x = np.squeeze(x)
    cs = ax.contourf(x, levels=shear_bounds,colors=shear_colors,extend='both')
    plt.title('ML Shear Max Swath (1/s)')
    plt.colorbar(cs, ticks=shear_bounds)
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])

    plt.ylabel('y (1/2 km)')
    
    plt.subplot(425)
    ax = plt.gca()
    x = ins[idx][5].reshape(60,60,1)
    x=np.squeeze(x)
    cs = ax.contourf(x,levels=MESH_bounds,colors=MESH_colors,extend='both')
    plt.axis('off')
    plt.title('MESH Swath 30 min prior (mm)')
    plt.colorbar(cs, ticks=MESH_bounds)
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])

    # plt.ylabel('y (1/2 km)')
    
    plt.subplot(426)
    ax = plt.gca()
    x = np.squeeze(ins[idx][0].reshape(60,60,1))
    cs = ax.contourf(x,levels=ref_levels,colors=ref_colors,extend='both')
    plt.title('Composite Max Swath (dBZ)')
    plt.colorbar(cs, ticks=ref_levels)
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])

    plt.ylabel('y (1/2 km)')

    plt.subplot(427)
    ax = plt.gca()
    x = np.squeeze(ins[idx][10].reshape(60,60,1))
    cs = ax.contourf(x,levels=wind_levels,colors=wind_colors,extend='both')
    plt.title('U-Wind Mean 0-6km (m/s)')
    plt.colorbar(cs, ticks=wind_levels)
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])

    plt.ylabel('y (1/2 km)')
    plt.xlabel('x (1/2 km)')

    plt.subplot(428)
    ax = plt.gca()
    x = np.squeeze(ins[idx][11].reshape(60,60,1))

    cs = ax.contourf(x,levels=wind_levels,colors=wind_colors,extend='both')
    plt.title('V-Wind Mean 0-6km (m/s)')
    plt.colorbar(cs, ticks=wind_levels)
    plt.xticks([0,10,20,30,40,50,60])
    plt.yticks([0,10,20,30,40,50,60])
    
    plt.ylabel('y (1/2 km)')
    plt.xlabel('x (1/2 km)')

    # wind profile? 

def plot(file,field="MESH_Max_30min",xlim=None,ylim=None):
    
    print(file)
    f = Dataset(file,mode='r')
    var = f.variables[field][:,:] # read in field
    var = np.where(var<-1000,0,var)
    var = np.squeeze(var)

    cs = ax.contourf(x,levels=ref_levels,colors=ref_colors,extend='both')
    plt.title('Composite Max Swath (dBZ)')
    plt.colorbar(cs, ticks=ref_levels)
    plt.xlabel('x (0.5 km)')
    plt.ylabel('y (0.5 km)')
    plt.colorbar()
    plt.savefig('noshear_test.png')

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

def check_accuracy(r,scaler,group='testing'):
    y_true = r['true_testing']
    y_pred = r['predict_testing']
    maxs = []
    pred_maxs = []
    scaler = scaler

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


def binary_accuracy(r,scaler,group='testing'):
    y_true = r['true_testing']
    y_pred = r['predict_testing']

    # scale data back to mm
    correct = 0
    FP = 0
    total = 0
    TP = 0
    TN = 0
    FN = 0
    events = 0
    for idx, ex in enumerate(y_true):
        ex = ex.reshape(1, 60*60)
        ex_pred = y_pred[idx].reshape(1,60*60)
        ex = scaler.inverse_transform(ex)
        ex_pred = scaler.inverse_transform(ex_pred)

        # Count Area Shared Above threshold (20 mm)
        thres = 20
        true_area = 0
        pred_area = 0
        common_area = 0
        for idx, row in enumerate(ex):
            for idx2, pixel in enumerate(row):
                if pixel >= thres:
                    true_area+=1
                    if ex_pred[idx][idx2] >= thres:
                        common_area+=1
                if ex_pred[idx][idx2] >= thres:
                    pred_area+=1
        
        # binarize the data
        ex_pred = np.where(ex_pred < 20, 0, 1) 
        ex = np.where(ex < 20, 0, 1)

        # Compute positives, negatives, etc
        for idx, row in enumerate(ex):
            for idx2, pixel in enumerate(row):
                pred_pix = ex_pred[idx][idx2]
                if pixel == pred_pix:
                    correct+=1
                    if pixel == 1:
                        TP+=1
                if pixel == 1:
                    events+=1
                if pixel != pred_pix and pred_pix == 1:
                    FP +=1
                if pixel == pred_pix and pred_pix ==0:
                    TN+=1
                if pixel != pred_pix and pred_pix == 0:
                    FN+=1
                total+=1
    print('correct total TP FP TN FN events, true area    predicted area    commmon area')
    return [correct, total, TP, FP, TN, FN, events], [true_area, pred_area, common_area]

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
