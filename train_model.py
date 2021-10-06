import numpy as np
import pandas as pd
import tensorflow as tf
import os
import fnmatch
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import random, argparse
import re
import png
import sklearn.metrics
import os
import fnmatch
import re
import numpy as np
from tensorflow.python.ops.numpy_ops.np_math_ops import true_divide
from unet import *
from job_control import *
import sys
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



# gpus = tf.config.experimental.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)
print('hi')
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def create_parser():
    parser = argparse.ArgumentParser(description='Hail Swath Learner')
    parser.add_argument('-exp_type',type=str,default='minimum',help='How to name this model?')
    parser.add_argument('-thres', type=float, default=0.25, help='Enter the dropout rate.' )
    parser.add_argument('-dropout', type=float, default=None, help='Enter the dropout rate.' )
    parser.add_argument('-lambda_regularization', type=float, default=0.1, help='Enter l1, l2, or none.') 
    parser.add_argument('-epochs', type=int, default=320, help='Training epochs')
    parser.add_argument('-results_path', type=str, default='~/MESH_results', help='Results directory')
    parser.add_argument('-lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('-patience', type=int, default=100 , help="Patience for early termination")   
    parser.add_argument('-network',type=str,default='unet',help='Enter u-net.')
    parser.add_argument('-unet_type', type=str, default='concat', help='Enter whether to concatenate or add during skips in unet')
    parser.add_argument('-filters',type=int, default=[16,32,64], help='Enter the number of filters for convolutional network')
    parser.add_argument('-batch_size',type=int, default=50, help='Enter the batch size.')
    parser.add_argument('-activation',type=str, default='relu', help='Enter the activation function.')
    parser.add_argument('-optimizer',type=str, default='adam', help='Enter the optimizer.')
    parser.add_argument('-kind',type=str,default='raw_Min',help='How is the data cleaned? (norm, binary, multiclass)')
    parser.add_argument('-exp_index', nargs='+', type=int, help='Array of integers')
    parser.add_argument('-type',type=str,default='regression',help='How type')
    parser.add_argument('-error',type=str,default='mse')
    return parser

def augment_args(args):
    # if you specify exp index, it translates that into argument values that you're overiding
    '''
    Use the jobiterator to override the specified arguments based on the experiment index. 
    @return A string representing the selection of parameters to be used in the file name
    '''
    index = args.exp_index
    if(index == -1):
        return ""
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    # Overides Ntraining and rotation
    if args.lambda_regularization != None:
        p = {'lambda_regularization: [0.0001, 0.005, 0.01]',
             'activation: ["elu","sigmoid","tanh","relu"]',
             'optimizer: ["adam","RMSProp","SGD-momentum"]'}
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    # destructively modifies the args 
    # string encodes info about the arguments that have been overwritten
    return ji.set_attributes_by_index(args.exp_index, args)

def transform(var):
    n_channels=var.shape[1]
    tdata_transformed = np.zeros_like(var)
    channel_scalers = []

    for i in range(n_channels):
        mmx = StandardScaler()
        slc = var[:, i, :, :].reshape(var.shape[0], 60*60) # make it a bunch of row vectors
        transformed = mmx.fit_transform(slc)
        transformed = transformed.reshape(var.shape[0], 60,60) # reshape it back to tiles
        tdata_transformed[:, i, :, :] = transformed # put it in the transformed array
        channel_scalers.append(mmx) # store the transform
                
    return tdata_transformed, channel_scalers

def transform_test(var,scalers):
    n_channels=var.shape[1]
    tdata_transformed = np.zeros_like(var)

    for i in range(n_channels):
        slc = var[:, i, :, :].reshape(var.shape[0], 60*60)
        transformed = scalers[i].transform(slc)
        transformed = transformed.reshape(var.shape[0], 60,60) # reshape it back to tiles
        tdata_transformed[:, i, :, :] = transformed # put it in the transformed array
    return tdata_transformed


def training_set_generator_images(ins, outs, batch_size=10,
                          input_name='input', 
                        output_name='output'):
    '''
    Generator for producing random minibatches of image training samples.
    
    @param ins Full set of training set inputs (examples x row x col x chan)
    @param outs Corresponding set of sample (examples x nclasses)
    @param batch_size Number of samples for each minibatch
    @param input_name Name of the model layer that is used for the input of the model
    @param output_name Name of the model layer that is used for the output of the model
    '''
   
    while True:
        # Randomly select a set of example indices
        example_indices = random.choices(range(ins.shape[0]), k=batch_size)
        
        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield({input_name: ins[example_indices,:,:,:]},
             {output_name: outs[example_indices,:,:,:]})

parser = create_parser()
args = parser.parse_args()

#ins,outs = load_data(cases_df, 451)
# ins_raw ins_raw_multi ins_raw_noNSE
ins = np.load('ins_{}.npy'.format(args.kind))
#np.save('ins_raw_no3D.npy',ins[:,:21,:,:])
outs = np.load('outs_raw.npy'.format(args.kind))
indices = np.asarray(range(ins.shape[0]))

ins_train, ins_test , outs_train, outs_test = train_test_split(ins, outs, test_size=0.25, random_state=3)
ins_train_indices, ins_test_indices , outs_train_indices, outs_test_indices = train_test_split(indices, indices, test_size=0.25, random_state=3)

#ins_train, ins_val, outs_train, outs_val = train_test_split(ins_train, outs_train, test_size =0.2, random_state=2)
#ins_train_indices, ins_val_indices, outs_train_indices, outs_val_indices = train_test_split(ins_train_indices, ins_train_indices, test_size =0.2, random_state=2)

# scaling
ins_train, scalers = transform(ins_train)
# ins_val = transform_test(ins_val,scalers)
ins_test = transform_test(ins_test,scalers)


outs_train, scalers = transform(outs_train)
# outs_val = transform_test(outs_val,scalers)
outs_test = transform_test(outs_test,scalers)

# save 
pickle.dump(scalers, open('scaler_{}.pkl'.format(args.kind),'wb'))

with strategy.scope():
    model = create_uNet(ins_train.shape[1:], nclasses=5,lambda_regularization=args.lambda_regularization,
                        activation=args.activation, dropout=args.dropout,
                        type=args.type, optimizer=args.optimizer,threshold=args.thres)
model.summary() 

# experiment with smaller batch sizes, as large batches have smaller variance
generator = training_set_generator_images(ins_train, outs_train, batch_size=args.batch_size,
                        input_name='input',
                        output_name='output')

early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                    restore_best_weights=True,
                                                    min_delta=0.0)
# Learn
# history = model.fit(x=generator, 
#                     epochs=epochs, 
#                     steps_per_epoch=10,
#                     use_multiprocessing=False, 
#                     validation_data=(ins_val, outs_val),
#                     verbose=True)

history = model.fit(x=generator, 
                    epochs=args.epochs, 
                    steps_per_epoch=10,
                    use_multiprocessing=False, 
#                    validation_data=(ins_val, outs_val),
                    verbose=True, 
                    callbacks=[early_stopping_cb])

model.compile(loss=my_MSE_fewer_misses, metrics='mse')

history = model.fit(
[x_train_vis,x_train_ir], y_train,
validation_data=([x_test_vis,x_test_ir], y_test),
epochs=100, batch_size=10)


results = {}
#results['args'] = args
results['true_outs'] = outs
results['predict_training'] = model.predict(ins_train)
results['predict_training_eval'] = model.evaluate(ins_train, outs_train)
results['true_training'] = outs_train
#results['predict_validation'] = model.predict(ins_val)
#results['predict_validation_eval'] = model.evaluate(ins_val, outs_val)
#results['true_validation'] = outs_val
results['true_testing'] = outs_test
results['predict_testing'] = model.predict(ins_test)
results['predict_testing_eval'] = model.evaluate(ins_test, outs_test)
results['outs_test_indices'] = outs_test_indices
#results['folds'] = folds
results['history'] = history.history

# Save results
fbase = r"results/{}_ins_{}_epochs_{}_multi_dropout_{}_batchsize_{}_{}_l2_{}_opt_{}_{}_thres_{}".format(args.exp_type,args.kind,args.error,args.dropout,args.batch_size,args.unet_type, args.lambda_regularization,args.optimizer,args.type,args.thres)
results['fname_base'] = fbase
fp = open("%s_results.pkl"%(fbase), "wb")
pickle.dump(results, fp)
fp.close()

# Model
model.save("%s_model"%(fbase))

print(fbase)