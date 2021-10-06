#from load_data import DATA_HOME
import pandas as pd 
import numpy as np
import os, sys, datetime

os.system("cp test.py rest.py")

path_to_localmax = "/csv/localmax/"
# loop through bottom and top of the hour for all cases

rootdir = '/mnt/data/SHAVE_cases'
DATA_HOME='/mnt/data/SHAVE_cases'
OUT_HOME='/mnt/data/michaelm/practicum/cases'
delta = 0.15
degrees = ['01.00','02.00','03.00','04.00','05.00','06.00','07.00','08.00','09.00','10.00','11.00','12.00','13.00','14.00','15.00','16.00','17.00','18.00','19.00','20.00']
pd.set_option("display.max_rows", None, "display.max_columns", None)

# gather cases
def get_cases(): 
    ref_count = 0
    total = 0
    ins_count=0
    cases_df = pd.DataFrame(columns={'casedate','multi_n','storms'})
    casedirs = []
    for subdir, dirs, files in os.walk(OUT_HOME):
        idx=0
        for dir in sorted(dirs):
            if dir[:2] == '20':
                casedate = dir
                for subdir, dirs, files in os.walk('{}/{}'.format(OUT_HOME,dir)):
                    for dir in dirs:
                        if dir[:5] == 'multi':
                            multi_n = dir[5]
                            accept = False
                            storms = []
                            for subdir, dirs2, files in os.walk('{}/{}/{}'.format(OUT_HOME,casedate,dir)):
                                for d in dirs2:
                                    if d[:5] == 'storm':
                                        for subdir, dirs3, files in os.walk('{}/{}/{}/{}'.format(OUT_HOME,casedate,dir,d)):
                                            for d3 in dirs3:
                                                if d3 == 'Reflectivity_-10C_Max_30min':
                                                    accept = True
                                                    storm_n = d[5:]
                                                    storms.append(storm_n)
#                                                     if accept == True:
#                                                         total+=1
#                                                         for subdir, dirs4, files in os.walk('{}/{}/{}/{}'.format(OUT_HOME,casedate,dir,d)):
#                                                             for d4 in dirs4:
#                                                                 if d4 == 'target_MESH_Max_30min':
#                                                                     ref_count+=1
# #                                                    print(ref_count,total,casedate,multi_n)
                                                    ins_count+=1
                                                    break
                            if accept == True:
                                row = {'casedate':casedate,'multi_n':multi_n,'storms':storms}
                                cases_df.loc[idx] = row
                                idx+=1
        break
    print(ins_count)
    return cases_df

# def clean():
#     for subdir, dirs, files in os.walk(OUT_HOME):
#         idx=0
#         for dir in dirs:
#             if idx == 1:
#                 sys.exit()
#             idx=+1
#             if dir[:2] == '20':
#                 casedate = dir
#                 print(casedate)
#                 for subdir, dirs, files in os.walk('{}/{}'.format(OUT_HOME,dir)):
#                     for d1 in dirs:
#                         if d1[:5] == 'multi':
#                             multi_n = d1[5]
#                             for subdir, dirs2, files in os.walk('{}/{}/{}'.format(OUT_HOME,casedate,d1)):
#                                 for d2 in dirs2:
#                                     if d2[:5] == 'storm':
#                                         for subdir, dirs3, files in os.walk('{}/{}/{}/{}'.format(OUT_HOME,casedate,d1,d2)):
#                                             for d3 in sorted(dirs3):
#                                                 if d3 == 'Reflectivity_-10C_Max_30min':
#                                                     for subdir, dirs, files2 in os.walk('{}/{}/{}/{}/{}/01.00'.format(OUT_HOME,casedate,d1,d2,d3)):
#                                                         for f2 in sorted(files2):
#                                                             if f2[-10:-7] == '000':
#                                                                 timestep = sorted(files2)[0][-13:-7]
#                                                             else:
#                                                                 os.system('rm {}/{}/{}/{}/{}/01.00/{}'.format(OUT_HOME,casedate,d1,d2,d3,f2))
#                                                 if d3 == 'MergedReflectivityQC':
#                                                     for deg in degrees:
#                                                         for subdir, dirs4, files3 in os.walk('{}/{}/{}/{}/{}/{}'.format(OUT_HOME,casedate,d1,d2,d3,deg)):
#                                                             for f3 in files3:
#                                                                 if f3[-13:-7] != timestep:
#                                                                     os.system('rm {}/{}/{}/{}/{}/{}/{}'.format(OUT_HOME,casedate,d1,d2,d3,deg,f3))
#                                                 elif d3 == 'target_MESH_Max_30min':
#                                                     for subdir, dirs5, files5 in os.walk('{}/{}/{}/{}/{}/MESH_Max_30min/01.00'.format(OUT_HOME,casedate,d1,d2,d3)):
#                                                         for f5 in files5:
#                                                             if f5[-10:-7] != '000':
#                                                                 os.system('rm {}/{}/{}/{}/{}/MESH_Max_30min/01.00/{}'.format(OUT_HOME,casedate,d1,d2,d3,f5))
#                                                 else:
#                                                     for subdir, dirs, files2 in os.walk('{}/{}/{}/{}/{}/01.00'.format(OUT_HOME,casedate,d1,d2,d3)):
#                                                         for f2 in sorted(files2):
#                                                             if f2[-10:-7] != '000':
#                                                                 os.system('rm {}/{}/{}/{}/{}/01.00/{}'.format(OUT_HOME,casedate,d1,d2,d3,f2)) 
#         break


# given a date, retrieve positions for max ref storm
def get_storm_info(date,multi_n):

    LOCALMAX_PATH = '{}/{}/multi{}/csv/'.format(OUT_HOME,date,multi_n)
    case_df = pd.DataFrame(columns={"timedate","Latitude","Longitude","Storm","Reflectivity"})
    i=+1
    with open('{}/{}/finished{}'.format(DATA_HOME,date,multi_n)) as f:
        for line in f:
            latN_d, lonW_d, latS_d, lonE_d = line.split()
            latN_d = float(latN_d)
            lonW_d = float(lonW_d)
            latS_d = float(latS_d)
            lonE_d = float(lonE_d)            
            break
    f.close()

    # builds dataframe of case centers 
    for subdir, dirs, files in os.walk(LOCALMAX_PATH):
        files = sorted(files)
        length=len(sorted(files))-1
        # Loop through times
        for idx, file in enumerate(files):
            # build dataframe of locations
            timedate=file[-19:-4]
            minutes = timedate[-4:-2]
            if (minutes == '30' or minutes == '00') and idx != 0 and idx != length:
                print("Finding localmax for ", timedate)
                df = pd.read_csv('{}/MergedReflectivityQCCompositeMaxFeatureTable_{}.csv'.format(LOCALMAX_PATH, timedate))
                if df.empty:
                    print("Empty dataframe!")  
                else:
                    # List of valid clusters
                    valid_clusters = {}
                    keys = range(df.shape[0])
                    for i in keys:
                        valid_clusters[i] = True
                    # find max
                    for idx, val in enumerate(df["MergedReflectivityQCCompositeMax"]):
                        if valid_clusters[idx] == False:
                            continue
                        if val < 40 or df['Size'].iloc[idx] < 20:
                            valid_clusters[idx] = False
                            continue
                        lat = df['#Latitude'].iloc[idx]
                        lon = df['Longitude'].iloc[idx]
                        latN = lat + delta
                        latS = lat - delta
                        lonW =  lon - delta
                        lonE =  lon + delta
                        # Don't include clusters too close to domain edge
                        if latN > (latN_d - 0.16) or latS <= (latS_d + 0.16) or lonW < (lonW_d + 0.16) or lonE >= (lonE_d-0.16):
                            valid_clusters[idx] = False
                            continue
                        for idx2, val2 in enumerate(df["MergedReflectivityQCCompositeMax"]):
                            if idx2 == idx or valid_clusters[idx2] == False:
                                continue
                            if df['Size'].iloc[idx2] < 20 or val2 < 40:
                                valid_clusters[idx2] = False
                                continue 
                            lat2 = df['#Latitude'].iloc[idx2]
                            lon2 = df['Longitude'].iloc[idx2]
                            if lat2 < latN and lat2 > latS and lon2 > lonW and lon2 < lonE:
                                if val2 > val: 
                                    valid_clusters[idx] = False
                                else:
                                    valid_clusters[idx2] = False
                    # valid_clusters is complete
                    # # add valid rows to case_dfS
                    for key in valid_clusters.keys():
                        if valid_clusters[key] == False:
                            continue
                        else:
                            row_idx = key
                            try:
                                row = {"timedate":timedate,"Latitude":df['#Latitude'].iloc[row_idx],"Longitude":df['Longitude'].iloc[row_idx],'Storm':df['RowName'].iloc[row_idx],'Reflectivity':df['MergedReflectivityQCCompositeMax'].iloc[row_idx]}
                            except:
                                print(row_idx)
                            case_df.loc[len(case_df.index)] = row
    case_df = case_df.sort_values(['timedate'])
    return case_df

# Build swaths
def w2accumulator(case_df, multi_n, fields):
    date = case_df['casedate']
    multi_n = case_df['multi_n']
    multi = '{}/{}/multi{}'.format(OUT_HOME,date,multi_n)
    os.system('rm -r {}/Mer* {}/ME* {}/Re* {}/tar*'.format(multi,multi,multi,multi))
    os.system('makeIndex.pl {}/{}/multi{} code_index.xml'.format(DATA_HOME,date,multi_n))
    for field in fields:
        os.system('w2accumulator -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -g {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/uncropped -C 1 -t 30 --verbose="severe"'.format(date, multi_n, field,date, multi_n))
        if field[8:] == 'Shear':
            os.system('w2accumulator -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -g {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/uncropped -C 3 -t 30 --verbose="severe"'.format(date, multi_n, field,date, multi_n))

def get_NSE(date,multi_n,fields):
    for field in fields:
        os.system('mkdir {}/{}/NSE'.format(OUT_HOME,date))
        os.sytem('ln -s /mnt/data/SHAVE_cases/{}/multi{}/NSE/{} /mnt/data/michaelm/practicum/cases/{}/NSE'.format(date,multi_n,field,date))

# Run localmax on composite reflectivity
def localmax(date, multi_n):
    os.system('makeIndex.pl /mnt/data/SHAVE_cases/{}/multi{} code_index.xml'.format(date,multi_n))
    os.system('w2localmax -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -I MergedReflectivityQCComposite -o /mnt/data/michaelm/practicum/cases/{}/multi{} -s -d "40 60 5"'.format(date,multi_n,date,multi_n))
    os.system('makeIndex.pl /mnt/data/michaelm/practicum/cases/{}/multi{} code_index.xml'.format(date,multi_n))
    os.system('w2table2csv -i {}/{}/multi{}/code_index.xml -T MergedReflectivityQCCompositeMaxFeatureTable -o {}/{}/multi{}/csv -h'.format(OUT_HOME,date,multi_n,OUT_HOME,date,multi_n))


def cropconv(case_df, date, nse_fields, fields_accum, multi_n):    
    os.system('makeIndex.pl {}/{}/NSE code_index.xml'.format(DATA_HOME,date))
    for idx, row in case_df.iterrows():
        # if idx <= 200:
        #     continue
        multi = '{}/{}/multi{}'.format(OUT_HOME,date,multi_n)
        lon = row['Longitude']
        lat = row['Latitude']
        delta = 0.15

        lonNW = lon - delta
        latNW = lat + delta
        lonSE = lon + delta
        latSE = lat - delta
        
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=1)).strftime('%Y%m%d-%H%M%S')

        # crop input
        #########################
        os.system("makeIndex.pl {}/{}/multi{}/uncropped code_index.xml {} {}".format(OUT_HOME,date,multi_n, time1, time2)) # make index for uncropped
        for field in fields_accum:
           os.system('w2cropconv -i {}/{}/multi{}/uncropped/code_index.xml -I {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,date, multi_n, field, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
        #########################

        # crop target -
        #########################
        time1 = (date_1+datetime.timedelta(minutes=30)).strftime('%Y%m%d-%H%M%S')
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=1)).strftime('%Y%m%d-%H%M%S')        
        os.system("makeIndex.pl {}/{}/multi{}/uncropped code_index.xml {} {}".format(OUT_HOME,date,multi_n, time1, time2))
        os.system('w2cropconv -i {}/{}/multi{}/uncropped/code_index.xml -I MESH_Max_30min -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d}/target_MESH_Max_30min -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,date, multi_n, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
        # ########################

        # NSE 
        # crop for 30 min prior to 30 min ahead
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time1 = (date_1+datetime.timedelta(minutes=-30)).strftime('%Y%m%d-%H%M%S')
        time2 = (date_1+datetime.timedelta(minutes=30)).strftime('%Y%m%d-%H%M%S')

        os.system("makeIndex.pl {}/{}/NSE code_index.xml {} {}".format(DATA_HOME,date, time1, time2))
        for field in nse_fields:
            os.system('w2cropconv -i {}/{}/NSE/code_index.xml -I {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d}/NSE -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(DATA_HOME,date, field, date, multi_n, idx, latNW, lonNW,latSE,lonSE))

        # commenting this out for repair
        os.system('w2cropconv -i {}/{}/multi{}/code_index.xml -I  MergedReflectivityQC -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(DATA_HOME,date, multi_n, date, multi_n, idx, latNW, lonNW,latSE,lonSE))

fields = ['MergedLLShear','MergedMLShear','MESH','Reflectivity_0C','Reflectivity_-10C','Reflectivity_-20C', 'MergedReflectivityQCComposite']
fields_accum = ['MergedLLShear_Max_30min','MergedMLShear_Max_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10C_Max_30min','Reflectivity_-20C_Max_30min', 'MergedReflectivityQCComposite_Max_30min',
                'MergedLLShear_Min_30min','MergedMLShear_Min_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-50C']

#fields = []
#fields_accum = ['Reflectivity_-10C_Max_30min', 'Reflectivity_-20C_Max_30min']
from ast import literal_eval



# cases_df = pd.read_csv('{}/cases_.csv'.format(OUT_HOME))
# i=0
# summ=0
# for _, case in cases_df.iterrows():
#     i+=1
#     summ+=len(literal_eval(case['storms']))
# print(summ)

# CASE TO REMEMBER : 20080710

cases_df = get_cases()
cases_df.to_csv('{}/cases_df.csv'.format(OUT_HOME))
sys.exit()

#finished_df = pd.read_csv('{}/finished_df.csv'.format(OUT_HOME)).sort_values(['casedate'])
#cases_df = cases_df.sort_values(['casedate'])
cases_df = pd.read_csv('{}/cases.csv'.format(OUT_HOME))
cases_df = cases_df.sort_values(['casedate'])

allstorms_info_old = pd.DataFrame(columns={"timedate","Latitude","Longitude","Storm","Reflectivity"})
for idx, case in cases_df.iterrows():

    date = case['casedate'] 
    multi_n = case['multi_n']

    if date < 20080728:
        continue

    os.system('rm -r {}/{}/multi{}/storm*'.format(OUT_HOME,date,multi_n))

    # os.system('rm -r {}/{}/multi{}/NSE'.format(OUT_HOME,date,multi_n)) 

    # get swaths
    #w2accumulator(case, multi_n, fields)

    # run localmax
    #localmax(date,multi_n)

    # gather individual storm info
    storm_df = get_storm_info(date, multi_n)
    # allstorms_info_df.append(storm_df)
    cropconv(storm_df, date, NSE_fields, fields_accum, multi_n)

allstorms_info_old.to_csv('allstorms_old.csv')

