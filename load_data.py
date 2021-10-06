'''
Author: Michael Montalbano
Date: 05/26/2021
'''
import datetime
from ast import literal_eval
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import sys, os
from pandas._libs import missing
from pandas.core.indexes import multi
from sklearn.model_selection import train_test_split
from display import *
np.set_printoptions(threshold=sys.maxsize)

# ins = np.load('ins_filtered_noNSE.npy')
# sys.exit()

ins = np.load('ins_raw.npy')

for i in [np.arange(40,22,-1),np.arange(21,17,-1),np.arange(16,11,-1),np.arange(9,-1,-1)]:
    ins = np.delete(ins,i,axis=1)
# # sys.exit()
# ins = np.delete(ins,8,axis=1) # ref swath
# ins = np.delete(ins,6,axis=1) # ref 0
# ins = np.delete(ins,7,axis=1) # ref -10
# ins = np.delete(ins,0,axis=1) # ref -20
# ins = ins[:,0:20,:,:]
print(ins.shape)
np.save('ins_raw_Min.npy',ins)
sys.exit()
# ins_raw_Min delete np.arange(40,22,-1),np.arange(21,17,-1),np.arange(16,11,-1),np.arange(9,-1,-1)
# ins_raw_noShear ins_raw_noRef ins_raw ins_raw_noMRMS ins_raw_noNSE ins_raw_RefAndShearAndMESH ins_raw_noMESH ins_noHeight
# threshold for binarizing MESH
thres = 29

DATA_HOME = '/mnt/data/michaelm/practicum/cases/'
degrees = ['01.00','02.00','03.00','04.00','05.00','06.00','07.00','08.00','09.00','10.00','11.00','12.00','13.00','14.00','15.00','16.00','17.00','18.00','19.00','20.00']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km','Heightof0C','Heightof-20C','Heightof-50C']           # Read in outputs
multi_fields = ['MergedReflectivityQCComposite_Max_30min','MergedLLShear_Max_30min','MergedMLShear_Max_30min','MergedLLShear_Min_30min','MergedMLShear_Min_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min']
fields3D = ['MergedReflectivityQC']
products = multi_fields + fields3D
products=multi_fields+NSE_fields+fields3D
cases_df = pd.read_csv('{}/cases_df.csv'.format(DATA_HOME))
print(products[0:24])
print(len(multi_fields+NSE_fields))
sys.exit()
# scaler
scaler = open_pickle('scaler.pkl')
scaler = scaler[0]

cases_info = pd.DataFrame(columns={"casedate","multi_n","storm",'index'})
# pd.set_option('expand_frame_repr', True)
# df = pd.read_csv('missing_df.csv')
# for _, r in df.iterrows():
#     print(r['filename'])
# print(df)

def load_data(cases_df, n, categorize=False, binary=False, normalize=False):
    missing_df = pd.DataFrame(columns={'casedate','multi_n','filename'})
    ins_full = []
    outs_full = []
    l=0
    index=0
    for idx, case in cases_df.iterrows(): 
        if l > n:
            print('done')
            return ins_full, outs_full, cases_info
        
        casedate = case['casedate']
        multi_n = case['multi_n']
        storms = literal_eval(case['storms'])
        # print(casedate,multi_n,storms)
        # print(type(storms))
        l+=1
        if casedate == 20080626 and multi_n == 1:
            continue
        if casedate == 20120728 and multi_n == 0:
            continue
        if casedate == 20080604 and multi_n == 1:
            continue
        if casedate == 20080610 and multi_n == 1:
            continue
        if casedate == 20080619 and multi_n == 2:
            continue
        if casedate == 20080626 and multi_n == 0:
            continue
        if casedate == 20080714 and multi_n == 2:
            continue
        if casedate == 20080721 and multi_n == 4:
            continue
        # dummy product to move to navigate to times in directory
        product = 'MergedReflectivityQCComposite_Max_30min'
        ins = []
        print('entering storm cycle')
        for storm in storms:
            row = {"casedate":casedate,'multi_n':multi_n,'storm':storm,'index':index}
            index+=1
            cases_info.loc[len(cases_info.index)] = row
            i=0
            j=0
            k=0
            # find timestep
            FIELD_PATH = '{}/{}/multi{}/storm{}/MESH_Max_30min/01.00'.format(DATA_HOME, casedate, multi_n, storm, product)
            timestamp=0
            #print(FIELD_PATH)
            for subdir, dirs, files in os.walk(FIELD_PATH):
                for file in sorted(files):
                    if file[-10:-7] == '000':
                        timestamp = file[:-7]
            # cycle through products
            ins = []
            outs = []


            # Read in outputs
            ################################################
            outs = []
            #print(timestamp)
            date_1 = datetime.datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            timestamp2 = (date_1+datetime.timedelta(minutes=30)).strftime('%Y%m%d-%H%M%S')
            FIELD_PATH = '{}/{}/multi{}/storm{}/target_MESH_Max_30min/MESH_Max_30min/01.00'.format(DATA_HOME, casedate, multi_n, storm)
            file = '{}/{}.netcdf'.format(FIELD_PATH, timestamp2)
            try:
                f = Dataset(file, mode='r')
            except:
                print(file)
                print('there is nothing here')
                break
            var = f.variables['MESH_Max_30min'][:,:]
            var = np.where(var<-1000,0,var)
            if categorize:
                var = categorize_data(var,binary=binary)
            if normalize:
                var = normalize_data(var)
            outs.append(var)
            ################################################

            for ind, product in enumerate(products):
                # Process fields in multi directory
                if product in multi_fields:
                    i+=1
                    FIELD_PATH = '{}/{}/multi{}/storm{}/{}/01.00'.format(DATA_HOME, casedate, multi_n, storm, product)
                    file = '{}/{}.netcdf'.format(FIELD_PATH, timestamp)
                    try:
                        f = Dataset(file, mode='r')
                    except: 
                        print(file)
                        break
                    var = f.variables[product][:,:] # read in field
                    var = np.where(var<-50,0,var) # set very large negative values to 0
                    
                    # binarize MESH
                    if binary:
                        if product == 'MESH_Max_30min':
                            var = np.where(var>=thres,1,var)
                            var = np.where(var<thres,0,var)

                    # Max and Min for Shears
                    if product == 'MergedLLShear_Min_30min' or product == 'MergedMLShear_Min_30min':
                        var = np.where(var>0,0,var) # set values greater than 0 to 0
                    if product == 'MergedLLShear_Max_30min' or product == 'MergedMLShear_Max_30min':
                        var = np.where(var<0,0,var) # set values less than 0 to 0


                    ins.append(var)
            
#                 # Process fields in NSE directories
                if product in NSE_fields:
                    j+=1
                    nse_time = '{}0000.netcdf'.format(timestamp[:11]) # grab start of hour nseanalysis for both '00' and '30'
                    file = '{}/{}/multi{}/storm{}/NSE/{}/nseanalysis/{}'.format(DATA_HOME, casedate, multi_n, storm, product, nse_time)
                    try:
                        f = Dataset(file, mode='r')
                    except FileNotFoundError:
                        date_1 = datetime.datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
                        timestamp2 = (date_1+datetime.timedelta(hours=1)).strftime('%Y%m%d-%H%M%S')
                        nse_time = '{}0000.netcdf'.format(timestamp2[:11])
                        file = '{}/{}/multi{}/storm{}/NSE/{}/nseanalysis/{}'.format(DATA_HOME, casedate, multi_n, storm, product, nse_time)
                        try:
                            f = Dataset(file, mode='r')
                        except:
                            print('not present',file)
                            break
                    var = f.variables[product][:,:]
                    var = np.where(var<-100,0,var)
                    ins.append(var)
#                 # Process fields with more than 1 degree
                if product in fields3D:
                    for degree in degrees:
                        k+=1
                        FIELD_PATH = '{}/{}/multi{}/storm{}/{}/{}'.format(DATA_HOME, casedate, multi_n, storm, product, degree)
                        file = '{}/{}.netcdf'.format(FIELD_PATH, timestamp)
                        try:
                            f = Dataset(file, mode='r')
                        except:
                            print(file)
                            break
                        var = f.variables[product][:,:]
                        var = np.where(var<0,0,var)
                        ins.append(var)
            ins_full.append(ins)
            outs_full.append(outs)
    missing_df.to_csv('missing_df.csv')
    return ins_full, outs_full, cases_info

def categorize_data(var,binary=False):
    thres = 29
    if binary:
        var = np.where(var<thres,np.array(1,0,0,0,0),var)
        var = np.where(var>=thres,1,var)
    if not binary:
        var = np.where(var<10,[1,0,0,0,0],var)
        var = np.where(var<20,np.asarray([0,1,0,0,0]),var)
        var = np.where(var<35,np.asarray([0,0,1,0,0]),var)
        var = np.where(var<50,np.asarray([0,0,0,1,0]),var)
        var = np.where(var>=50,np.asarray([0,0,0,0,1]),var)
    return var
def normalize_data(data):
    # normalize data between 0 and s
    norm = np.max(data)
    if norm == 0:
        return data
    return data/norm

n=1000

#ins,outs,cases_info = load_data(cases_df, n, categorize=False, binary = False,normalize=False)

# ins,outs,cases_info = load_data(cases_df, n, categorize=True, binary = False,normalize=False)


# ins = np.asarray(ins)
# outs = np.asarray(outs)
# print(ins.shape)
# np.save('ins_v3.npy',ins)
# np.save('outs_v3.npy',outs)
#cases_info.to_csv('cases_info.csv')

# # ins = np.reshape(ins, (ins.shape[0],60,60,41))
# # outs = np.reshape(outs, (outs.shape[0],60,60,1))

# # # print(ins.shape,outs.shape)

# # # categorize = True
# np.save('ins_raw.npy',ins)
# np.save('outs_raw.npy',outs)

#ins = np.load('ins_v3.npy')
outs = np.load('outs_raw.npy')

def categorize(outs):
    new_outs = []
    for idx1, ex in enumerate(outs):

        #ex = categorize_data(ex,binary=False)
        ex = ex.reshape(1,3600)
        ex = scaler.inverse_transform(ex)
        ex = ex.reshape(60, 60, 1)
        ex = np.squeeze(ex)

        new_ex = []
        for idx2, row in enumerate(ex):
            new_row = [] # create a new row
            for pixel in row:
                if pixel < 10:
                    pixel = np.array([1,0,0])
                elif pixel < 25:
                    pixel = np.array([0,1,0])
                elif pixel > 60:
                    pixel = np.array([0,0,1])
                new_row.append(pixel)
            # shape of new_row is (60,5)
            new_ex.append(np.array(new_row))
        new_outs.append(np.array(new_ex))
    new_outs = np.asarray(new_outs)
    new_outs = np.swapaxes(new_outs,1,3)
    return new_outs

#new_outs = categorize(outs)
#print(new_outs.shape)
#np.save('outs_raw_categorical.npy',new_outs)

# MESH is index 17 for ins
# new_outs = []
# for idx, example in enumerate(outs):
#     example = categorize_data(example,True)
#     new_outs.append(example)
# print(outs.shape)
# new_outs = []
# for idx, example in enumerate(outs):
#     example = categorize_data(example,True)
#     new_outs.append(example)

# new_outs = np.asarray(new_outs)
# print(new_outs.shape)

# np.save('outs_binary.npy',outs)









    
# EXTRA CODE IN CASE THERE ARE MISSING FILES
                        # try:
                        #     f = Dataset(file, mode='r')
                        # except FileNotFoundError:
                        #     row = {'casedate':casedate,'multi_n':multi_n}
                        #     if n == 0:
                        #         missing_df.loc[n] = row
                        #         n+=1
                        #     if n != 0:
                        #         print(file)
                        #         if not ((missing_df['casedate'] == casedate) & (missing_df['multi_n'] == multi_n)).any():
                        #             missing_df.loc[n] = row
                        #             n+=1

