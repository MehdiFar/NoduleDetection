from __future__ import print_function

from numpy.random import seed
from tensorflow import set_random_seed
import keras
from keras.models import Sequential
from keras.layers import *
import numpy as np
import os
import math
from scipy import ndimage
from keras.models import load_model
from my_classes_raidb import DataGenerator
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import datetime
import Compute_Accuracy as CA
import pandas as pd



def split_test_fold_files(nodule_adds, non_nodule_adds, ratio):
	# inputs are two lists of addresses of 
	# positive (nodule_adds) and negative (non_nodule_adds) samples, and divide the list 
	# into four lists:
	# outputs: 
	# 1. list of positive samples as validation (nodule_valid_adds) 
	# 2. list of negative samples as validation (non_nodule_valid_adds) 
	# 3. list of positive samples as test (nodule_test_adds) 
	# 4. list of negative samples as test (non_nodule_test_adds)
	random.shuffle(nodule_adds) # shuffle positive sample addresses
	random.shuffle(non_nodule_adds) # shuffle negative sample addresses
	end_index = int(len(nodule_adds)*ratio) # Location to divide positive list into validation and test
	nodule_valid_adds = nodule_adds[:end_index]  
	nodule_test_adds = nodule_adds[end_index:]  
	end_index = int(len(non_nodule_adds)*ratio) # Location to divide negative list into validation and test
	non_nodule_valid_adds = non_nodule_adds[:end_index]
	non_nodule_test_adds = non_nodule_adds[end_index:]
	return nodule_test_adds, non_nodule_test_adds, nodule_valid_adds, non_nodule_valid_adds


def generate_partition_labels(input_dic):
	# reads all training and test file addresses and store them in variables that later fed to keras fit_generator function. 
	# outputs: 1. partition: a dictionary with keys='test','train','validation', each has the value equal to a list of addresses; e.g. paritition['train'] is a list of all training sample addresses.
	#          2. labels: A dictionary with keys = All samples (training, validation, and test) addresses and value is equal to label of that sample (0 or 1)
	#		   3. test_labels: A list of label for test samples
	#		   4. valid_labels: A list of label for validation samples
	files_dir = os.path.join(input_dic['data_path'], str(input_dic['input_shape'][0]) +'x'+ str(input_dic['input_shape'][1]) +'x'+ str(input_dic['input_shape'][2]))
	test_fold = input_dic['test_fold']
	negative_ratio = input_dic['negative_ratio']
	validation_ratio = input_dic['validation_ratio'] # what percentage of test fold would be used as validation set

	nodule_dir = files_dir + '/' + 'Nodule_Augmented/'  # postive samples directory for training
	non_nodule_dir = files_dir + '/' + 'Non-Nodule/' # negative samples directory for training
	nodule_files_complete_addres = [] #list of addresses of positive samples
	non_nodule_files_complete_addres = [] #list of addresses of negative samples

	if(not input_dic['phantom_trainOnly']):	# if we don't want to fully train on phantom data

		for i in [x for x in range(6) if x != test_fold]:	# iterate through all folds of training samples (all folds except testing fold) and add their addresses to positive and negative address lists
			print(i)
			nodule_fold_dir = nodule_dir + 'Fold' + str(i)  # navigate to positive fold(i)
			non_nodule_fold_dir = non_nodule_dir + 'Fold' + str(i) 	# navigate to negative fold(i)
			nodule_files = os.listdir(nodule_fold_dir) 	# list of all positive files addresses in fold i
			non_nodule_files = os.listdir(non_nodule_fold_dir) 	# list of all negative files addresses in fold i
			if(input_dic['1or2_inclusion']==False): 	#if we decided not to include nodules annotation by less than 3 radiologists in the training.
				subtract = np.load('/gpfs_projects/mohammadmeh.farhangi/DiskStation/LIDC/Mehdi_Summer2018/Luna_Scans/candidates_1or2Rad.npy').tolist()	#read the file that contain the patch number of samples that are annotation by 1 or 2 radialogists
				subtract2 = [x.decode('utf-8') for x in subtract]
				non_nodule_files = list(set(non_nodule_files) - set(subtract2))	# Exclude patche addresses specified by 1 or 2 radiologists from training samples



			nodule_files_complete_addres = nodule_files_complete_addres + [nodule_fold_dir + '/' + s for s in nodule_files] 	# add all the files in fold i to the training files positive list
			non_nodule_files_complete_addres = non_nodule_files_complete_addres + [non_nodule_fold_dir + '/' + s for s in non_nodule_files]	# # add all the files in fold i to the training files negative list

		if(input_dic['include_phantom']):	# decide to include phantom data in training or not
			phantom_add = input_dic['phantom_dir'] + str(input_dic['input_shape'][0]) +'x'+ str(input_dic['input_shape'][1]) +'x'+ str(input_dic['input_shape'][2]) + '/' + 'Nodule/' # phantom data address
			phantom_files = os.listdir(phantom_add) 	# list of all files in phantom directory

			csv_content = pd.read_csv(input_dic['phantom_csv_dir']).values 	# The csv file containing information about phantom nodules is read. It will be used to exclude phantom nodules greater or samller than a threshold from training
			noduleVol = csv_content[:,3]	#column 3 of spreadsheet stores the nodule volume information
			files_indices = (np.where(np.logical_and(noduleVol>=input_dic['phantom_noduleMinSize'], noduleVol<=input_dic['phantom_noduleMaxSize']))[0]+2).tolist()  # indices of phantom nodules in the spread sheet that are less than a maximum threshold and greater than a minimum threshold are stored in files_indices list

			files_indices = [str(i) + '.npy' for i in files_indices]	# phantom file name in the hard disk is the same as their row number in the spread sheet. so, to find the address location of nodules that satsify the min and max nodule size, we just add a .npy to the end of the file_indices.
			phantom_files = list(set(files_indices) & set(phantom_files))	# Since we could not extract all the samples in phantom spreadsheet, we compute an intersection of files in directory and the files that satisfy the size location.
			print("{} samples from phantom data added to the trainig".format(len(phantom_files)))

			nodule_files_complete_addres = nodule_files_complete_addres + [phantom_add + s for s in phantom_files]	# phantom address files are added to positive training samples.



	nodule_labels = [1] * len(nodule_files_complete_addres)	# label of positive samples is 1
	non_nodule_labels = [0] * len(non_nodule_files_complete_addres)	#lable of negative samples is 0
	train_labels = nodule_labels + non_nodule_labels

	files_complete_addres = nodule_files_complete_addres + non_nodule_files_complete_addres

	#################### Generate test and validation file addresses and labels ##################
	nodule_dir = files_dir + '/' + 'Nodule_CandList/'
	non_nodule_dir = files_dir + '/' + 'Non-Nodule/'
	
	nodule_fold_dir = nodule_dir + 'Fold' + str(test_fold)	# positive samples in test fold directory
	non_nodule_fold_dir = non_nodule_dir + 'Fold' + str(test_fold)	# negative samples in test fold directory
	nodule_files = os.listdir(nodule_fold_dir)	# list of positive files in test fold
	non_nodule_files = os.listdir(non_nodule_fold_dir)	# list of negative files in test fold
	test_nodule_files_complete_addres = [nodule_fold_dir + '/' + s for s in nodule_files]	# complete address of positive samples: dir+filename
	test_non_nodule_files_complete_addres = [non_nodule_fold_dir + '/' + s for s in non_nodule_files]	# complete address of negative samples: dir+filename


	test_nodule_files_complete_addres, \
	test_non_nodule_files_complete_addres,  \
	valid_nodule_files_complete_addres, \
	valid_non_nodule_files_complete_addres = split_test_fold_files(test_nodule_files_complete_addres, test_non_nodule_files_complete_addres,validation_ratio)  # divide the test fold into validation and test set

	test_nodule_labels = [1] * len(test_nodule_files_complete_addres)
	test_non_nodule_labels = [0] * len(test_non_nodule_files_complete_addres)
	test_labels = test_nodule_labels + test_non_nodule_labels

	valid_nodule_labels = [1] * len(valid_nodule_files_complete_addres)
	valid_non_nodule_labels = [0] * len(valid_non_nodule_files_complete_addres)
	valid_labels = valid_nodule_labels + valid_non_nodule_labels

	test_files_complete_addres = test_nodule_files_complete_addres + test_non_nodule_files_complete_addres
	valid_files_complete_addres = valid_nodule_files_complete_addres + valid_non_nodule_files_complete_addres

	whole_label = train_labels + test_labels + valid_labels

	whole_files = files_complete_addres + test_files_complete_addres + valid_files_complete_addres

	partition = {'train':files_complete_addres,'test':test_files_complete_addres, 'validation':valid_files_complete_addres}
	# print(partition['validation'])
	labels = dict(zip(whole_files, whole_label))	

	return partition, labels, test_labels, valid_labels




def build_LRCN(input_dic):
	# The function design the structure of network based on parameters define in the begining of the script, and returns the keras model
	# This function use the GRU units to capture sequential information. In order to test the others like LSTM, one should should find the function build_LRCN and change GRU to LSTM directly.
	convLayers_no = input_dic['numberOfLayers']
	convFilts_size = input_dic['filterSizeTable']
	convFilts_no = input_dic['numberOfConvFilts']
	input_shape = input_dic['input_shape']
	poolingType = input_dic['poolingType']
	dropout_Convs = input_dic['dropout_Convs']
	numberOfRNNUnits = input_dic['numberOfRNNUnits']
	dropout_RNN = input_dic['dropout_RNN']
	RNN_arch = input_dic['RNN_arch']

	numberOfFCLayers = input_dic['numberOfFCLayers']
	sizeofFCLayers = input_dic['sizeOfFCLayers']
	dropout_FC = input_dic['dropout_FC']
	init = input_dic['initial']

	input_shape = input_shape + (1,) # keras input shape is 4-D for 3-D patches, last dimension defines the channel number, while building the model this value is set to 1
	model = Sequential() # type of model is sequential. the other type of defining deep learning model in keras is functional, which is not used in this script

	model.add(TimeDistributed(Conv2D(convFilts_no[0], convFilts_size[0], kernel_initializer = init, padding='same', activation='relu'), input_shape= input_shape)) # time distributed is a wrapper that apply Conv2D to every sequence of the input (the first dimension of input is seen as the sequence). For example, if input is (16x48x48) the conv2d is applied to each 16 sequences.
	if(poolingType[0] == 'Max'):
		model.add(TimeDistributed(MaxPooling2D(2, 2)))
	if(poolingType[0] == 'Avg'):
		model.add(TimeDistributed(AveragePooling2D(2, 2)))

	for i in range(1,convLayers_no,1):	# add layers one after the other (Sequential keras model)
		if(dropout_Convs[i-1] > 0):	# add drop out first before adding the next layer
			model.add(Dropout(dropout_Convs[i]))
		model.add(TimeDistributed(Conv2D(convFilts_no[i], convFilts_size[i], kernel_initializer = init, padding='same', activation='relu')))
		if(poolingType[i] == 'Max'):
			model.add(TimeDistributed(MaxPooling2D(2, 2)))
		if(poolingType[i] == 'Avg'):
			model.add(TimeDistributed(AveragePooling2D(2, 2)))

	model.add(TimeDistributed(Flatten()))	# flattern the patches inside every sequence. e.g. out put is (16x2304) instead of (1x36864)
	if(dropout_Convs[convLayers_no-1] > 0):
		model.add(Dropout(dropout_Convs[convLayers_no-1]))	

	# for a detailed description about each of these RNN_arch please find the paramsDL['RNN_arch'] at top of the script.
	if(RNN_arch == 0):
		model.add(GRU(numberOfRNNUnits, return_sequences=False, dropout=dropout_RNN))
	if(RNN_arch == 1):
		model.add(GRU(numberOfRNNUnits, return_sequences=True, dropout=dropout_RNN)) 
		model.add(Flatten())
	if(RNN_arch == 2):
		model.add(GRU(numberOfRNNUnits, return_sequences=True, dropout=dropout_RNN))
		model.add(GlobalAveragePooling1D())
	if(RNN_arch == 3):
		model.add(GRU(numberOfRNNUnits, return_sequences=True, dropout=dropout_RNN))
		model.add(Permute((2,1)))
		model.add(TimeDistributed(Dense(1, kernel_initializer = init, activation = 'relu')))
		model.add(Flatten())
	if(RNN_arch == 4):
		model.add(Bidirectional(GRU(numberOfRNNUnits, return_sequences=False, dropout=dropout_RNN)))
	if(RNN_arch == 5):
		model.add(Bidirectional(GRU(numberOfRNNUnits, return_sequences=True, dropout=dropout_RNN)))
		model.add(Permute((2,1)))
		model.add(TimeDistributed(Dense(1, kernel_initializer = init, activation = 'relu')))
		model.add(Flatten())
	if(RNN_arch == 6):
		model.add(Bidirectional(GRU(numberOfRNNUnits, return_sequences=True, dropout=dropout_RNN)))
		model.add(GlobalAveragePooling1D())
		
	if(RNN_arch == -1):
		model.add(TimeDistributed(Flatten()))
		model.add((Conv1D(64,(2), dilation_rate=(1), kernel_initializer = init, padding='same', activation='relu')))
		model.add((Conv1D(64,(2), dilation_rate=(2), kernel_initializer = init, padding='same', activation='relu')))
		model.add((Conv1D(64,(2), dilation_rate=(4), kernel_initializer = init, padding='same', activation='relu')))
		print(model.summary())
		model.add(Flatten())

	for i in range(numberOfFCLayers):	# Loop over number of fully connected layers
		model.add(Dense(sizeofFCLayers[i], kernel_initializer = init, activation = 'relu'))
		if(dropout_FC[i] > 0):
			model.add(Dropout(dropout_FC[i]))

	model.add(Dense(2, kernel_initializer = init, activation='softmax'))	

	return model


def load_model_cust(input_dic):
	# load the model if it already exists, or call build_LRCN if the model does not exists.
	# if paramsDL['num_epochs'] is set to x, and a saved model is loaded, then the model will 
	# be trained for another x epochs, no matter how many iterations it has already been trained. 
	# output: Keras model
	model_path = input_dic['model_path'] + input_dic['model_name']
	loss = input_dic['loss']
	learning_rate = input_dic['learning_rate']
	momentum = input_dic['momentum']
	decay = input_dic['decay']
	nesterov = input_dic['nesterov']

	opt_method = input_dic['opt']
	if(opt_method == 'SGD'):
		opt = keras.optimizers.SGD(lr = learning_rate, momentum = momentum, decay = decay, nesterov = nesterov)

	if(opt_method == 'RMSprop'):
		opt = keras.optimizers.RMSprop(lr = learning_rate)

	if(opt_method == 'Adam'):
		opt = keras.optimizers.Adam(lr = learning_rate, decay = decay)

	metrics = input_dic['metrics']
	if os.path.isfile(model_path):
		print('A saved model was found, initializing now ...')
		model = load_model(model_path)
	else:
		if(not os.path.isdir(input_dic['model_path'])):
			os.makedirs(input_dic['model_path'])
		model = build_LRCN(input_dic)
		print('No model found, initializing randomly ...')
		model.compile(loss=loss, optimizer=opt, metrics=metrics)

	print(model.summary())
	return model


def main_function(input_dic, training_files_dic):
	dim_x = input_dic['input_shape'][1]
	dim_y = input_dic['input_shape'][2]
	dim_z = input_dic['input_shape'][0]


	partition = training_files_dic['partition']
	labels = training_files_dic['labels']
	cls_true_test = training_files_dic['cls_true_test']
	cls_true_valid = training_files_dic['cls_true_valid']

	batch_size = input_dic['batch_size']
	epochs = input_dic['num_epochs']
	class_weight = input_dic['class_weight']

	patience = input_dic['stoppingCriteria']

	if(patience > 0):	# if stopping criteria is based on changes in validation loss, use early stopping and save the best model
		call_backs = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto'),
				  keras.callbacks.ModelCheckpoint(filepath = input_dic['model_path'] + input_dic['model_name'], monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
	if(patience == -1):	# if stopping criteria is not based on changes in validation loss, do not include early stopping in callbacks.
		call_backs = [keras.callbacks.ModelCheckpoint(filepath = input_dic['model_path'] + input_dic['model_name'] + '_best', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)]
	model = load_model_cust(input_dic)

	############ Next lines for training ##################

	params = {'dim': input_dic['input_shape'],
			'batch_size': batch_size,
			'shuffle': True}
	training_generator = DataGenerator(partition['train'], labels, **params)    # DataGenerator is a function defined in my_classes_raidb Script. It prepares the data to be fed to keras fit_generator. For more informaton about this script and function visit: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
	validation_generator = DataGenerator(partition['validation'], labels, **params) # for more information about DataGenerator visit: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html


	history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=len(partition['train'])//batch_size,
                    validation_steps = len(partition['validation'])//batch_size,
                    use_multiprocessing=True,
                    max_queue_size=64,
                    class_weight=class_weight,
                    epochs=epochs,
                    workers=64,
                    callbacks = call_backs)	# fit_generator is the function that train the model.
	model.save(input_dic['model_path'] + input_dic['model_name'])	# save the last model model after the last epoch.

	############# next lines compute conf matrix on test data ############
	params = {'dim': input_dic['input_shape'],
			'batch_size': 1,
			'shuffle': False}	# first shuffle is turned off to labels matches the samples order.
	test_generator = DataGenerator(partition['test'], labels, **params)
	predict = model.predict_generator(test_generator,steps=len(partition['test']))
	conf_matrix = confusion_matrix(cls_true_test, np.argmax(predict,axis=1))
	history.history['conf_matrix'] = conf_matrix
	fpr, tpr, thresholds = metrics.roc_curve(cls_true_test, np.argmax(predict,axis=1), pos_label=1) #compute area under the curve

	auc = metrics.auc(fpr, tpr)
	history.history['auc'] = auc
	history.history['predict'] = predict
	history.history['cls_true'] = cls_true_test	# all the metrics on test set are stored in history variable and saved in the hard disk.

	min_valid_loss = min(history.history['val_loss'])
	np.save(input_dic['model_path'] + input_dic['model_name'] + '_epoch' + str(len(history.history['val_loss']) - patience).zfill(3) +  '_valLoss' + "{:.4f}".format(min_valid_loss) + '_auc' + "{:.4f}".format(auc) + '_history.npy', history.history)
	np.save(input_dic['model_path'] + input_dic['model_name'] + '_params.npy',input_dic)  # save history and paramsDL in the hard disk.


	CA.accuracy_one_fold(input_dic)	# CA is the script "ComputeAccuracy" which is written based on the Evaluation script provided by Luna challenge. This function predicts the score for samples in one fold and write the results in a csv output file
	CA.write_excel_files(input_dic)	# write the output results in a folder with the same name as model name. the generated files are FROC curve, bootstracp cvs file, and text file about CAD analysis, e.g. FP, TP, etc.
	CA.compute_CPM(input_dic,history.history)	# Compute CPM and write it in the analysis folder.

def main_function_iter(input_dic):

	partition, labels, cls_true_test, cls_true_valid = generate_partition_labels(input_dic)	# generate the validation, training and test file and labels by reading the files in 10 different folds.
	training_files_dic = {}
	training_files_dic['partition'] = partition
	training_files_dic['labels'] = labels
	training_files_dic['cls_true_test'] = cls_true_test
	training_files_dic['cls_true_valid'] = cls_true_valid	# The keras function fit_generator gets the train, validation, and test data as a dictionary and here the dictionary is constructed to be passed later on to keras function.

	main_function(input_dic,training_files_dic)


if __name__ == "__main__":
	############## Deep Learning parameters ##############
	# The initializations here can be seen as the default value for the hyper parameters.
	# At the end of the script and before calling the main function, these values can be changed.
	# So, one can keep these values as default and apply any changes in these parameters at 
	# the end of the script and before calling the main function, instead of changing the values here.

	paramsDL = {}	# All parameters of the routines in this script is defined and set inside the paramsDL dictionary. Inside each routine this variable is referred to as input_dic.

	paramsDL['optimized_param'] = 'NLST/FP_redux/Fold012345Only4Training/LUNAPatches/64x64x20'  # Subfoder where the model is saved, If this subfolder does not exist, it will be created. One may specifies folders in each subfolder using "/", for examples paramsDL['optimized_param'] = 'Y_30Epochs/Fold0'

	paramsDL['data_path'] = '/gpfs_projects/mohammadmeh.farhangi/shamidian/mehdi/patches/'       # location where patches will be stored.
	paramsDL['model_path'] = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/LIDC/Mehdi_Summer2018/Models/' + paramsDL['optimized_param'] + '/'  # If the model is trained from scrath, this parameter defines the location to save model parameters and hyperparameters. If we plan to do transfer learning (e.g. train on phantom and tune on Luna), This parameter defines the directory where the pre-trained model exists. 
	paramsDL['model_name'] = datetime.datetime.now().strftime("%I-%M%p_%B%d_%Y")	# The name of the model is generated based on the system time and date in order to keep it unique. If we plan to do transfer learning (e.g. train on phantom and tune on Luna) we need to specify the name of pretrained model here.
	paramsDL['test_fold'] = 9  # test fold on which the algorithm is evaluated. value 9 specifies that the network will be trained on folds 0,...,8 and tested on fold 9.
	paramsDL['1or2_inclusion'] = False  # when false, patches marked by 1 or 2 radiologists would not be included in training. When True they will.


	paramsDL['include_phantom'] = False # include phantom nodule samples in training or not; If true the phantom nodules with the size in the range of [paramsDL['phantom_noduleMinSize'], paramsDL['phantom_noduleMaxSize']] will be included to Luna positive samples.
	paramsDL['phantom_dir'] = '/gpfs_projects/mohammadmeh.farhangi/shamidian/mehdi/Phantom/'  # directory where phantom patches are stored
	paramsDL['phantom_csv_dir'] = '/gpfs_projects/mohammadmeh.farhangi/shamidian/mehdi/Phantom/phantomScanInfo.csv'	# spefcifies the location of csv file that contains the information of about 6,000 phantom nodules
	paramsDL['phantom_noduleMinSize'] = 0 # specifies the smallest phantom nodule size in order to be included in training. The minimum size of a nodule in phantom spreadsheet is 62.38 mm^3
	paramsDL['phantom_noduleMaxSize'] = 700 # specifies the biggest phantom nodule size in order to be included in training. The maximum size of a nodule in phantom spreadsheet is 34524.15 mm^3
	paramsDL['phantom_trainOnly'] = False 	# specifies to train only on phantom data or not. If True the network is trained on positive and negative patches extracted from phantom data. If False the train is done on Luna CT
										# negative and positive patches. In this case, phantom positive samples are included in training if the variable paramsDL['include_phantom'] is true.

	paramsDL['input_shape'] = (20,64,64)    # size of input patches 
	paramsDL['num_epochs'] = 30	# max number of epochs. The algorithm might stops before this value reached depending on the stopping criteria on validation set.
	paramsDL['batch_size'] = 32	# number of samples in each batch
	paramsDL['negative_ratio'] = 1	# percentage of negative samples that is used for training and testing
	paramsDL['validation_ratio'] = 0.3   # percentage of samples in test fold that will be used for validation

	paramsDL['learning_rate'] = 0.0001   # Learning rate is used when the optimizer is SGD or Adam, or RMSprop
	paramsDL['momentum'] = 0.9	# Momentum is only applied when the optimizer is SGD. Adam and RMSprop has their own default momentum which cannot be changed.
	paramsDL['decay'] = 1e-6	# Decay is only applied when the optimizer is SGD. Adam and RMSprop has their own default momentum which cannot be changed.
	paramsDL['initial'] = keras.initializers.glorot_uniform()  #default: glorot_uniform(), other options: glorot_normal(), lecun_uniform(), lecun_normal(), he_normal(), he_uniform(), Orthogonal(), VarianceScaling(), TruncatedNormal(), RandomUniform(), RandomNormal(), Constant(value=0), Zeros(), Identity(gain=1.0)
	paramsDL['class_weight'] = {0:1,1:8} # when the number of smaples in classes are unbiassed this metric might help faster convergence.
	paramsDL['loss'] = 'binary_crossentropy'
	paramsDL['opt'] = 'Adam'    #options: 'RMSprop', 'Adam', 'SGD'
	paramsDL['metrics'] = ['accuracy']
	paramsDL['nesterov'] = False # Only is used when the optimizer is SGD
	paramsDL['stoppingCriteria'] = -1 # Defines when to stop training, positive values indicate number of iterations to wait without imporovment in validation loss. -1 means stop after paramsDL['num_epochs'], no matter how validation loss changes. 

	paramsDL['numberOfLayers'] = 4 #number of conv layers , The value here should match the size of the next 4 lists: paramsDL['filterSizeTable'], paramsDL['numberOfConvFilts'], paramsDL['poolingType'], paramsDL['dropout_Convs']
	paramsDL['filterSizeTable'] = ((5,5), (3,3), (3,3), (3,3)) #one row per conv layer, so number of rows should match number of conv layers!
	paramsDL['numberOfConvFilts'] = [64,64,64,64]  # number of filters in each layer. number of entries should match the number of layers.
	paramsDL['poolingType'] = ['None','Max','Max','Max'] # Pooling that is used after each conv layer. The size should be the same as paramsDL['numberOfLayers']. Anything other than 'Max' and 'Avg' (e.g. 'None') means pooling does not apply at this layer
	paramsDL['dropout_Convs'] = [-1,-1,-1, 0.5]  # drop out rate that comes after each conv layer. -1 means no dropout

	paramsDL['RNNType'] = 'GRU'    # The type of RNN used, This parameter is not used anywhere in the code and just initialized to know which RNN is used in the script. In order to test the others like LSTM, one should should find the function build_LRCN and change GRU to LSTM directly.
	paramsDL['numberOfRNNUnits'] = 256
	paramsDL['dropout_RNN'] = 0.8	# drop in RNN unit
	paramsDL['RNN_arch'] = -1   # options: 0, 1, 2, 3, 4, 5   0: one output for every unit, 1: sequence number output for every unit. 2: average of sequences for every unit
	#														 3: weighted average of sequences for every unit, 4: one output for every unit and bidirectional GRU, 
	#														 5: weighted average of sequences for every unit and biderectional GRU
	#														 note: When bidirectional is used, it is better to keep numberofRNNUnits half the sizeOfFCLayers because bidirectional generates 2 output for every RNN unit

	paramsDL['numberOfFCLayers'] = 2 #number of FC layers
	paramsDL['sizeOfFCLayers'] = [128,128] # number of nodes in each FC layer. Number of entries should match paramsDL['numberOfFCLayers']
	paramsDL['dropout_FC'] = [0.7,0.7]  # drop out rate that comes after each FC layer, -1 means no dropout. Number of entries should match paramsDL['numberOfFCLayers']


	main_function_iter(paramsDL)	# The main function is defined as iterations, because one may want to change a hyper-parameter in a for loop in the future, but for now the main function is called just once inside this function.

