import numpy as np
import os
from keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from my_classes_raidb import DataGenerator
import csv
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogFormatter,StrMethodFormatter,FixedFormatter
import sklearn.metrics as skl_metrics
from NoduleFinding import NoduleFinding
from tools import csvTools
import random

def write_csv(input_dic, row_list, predict):
	# input: 1) input_dic, 2) row_list: rows in the candidate csv file, which will be filled in this function; rows that belong to the testing fold.
	# write the csv output file for only one fold
	# the file is written in a subfolder in the directory which the model is saved.
	model_path = input_dic['model_path']+input_dic['model_name']
	Generated_files_dir = model_path + '_csvResults/' # subfolder which the csv file will be saved
	Generated_CSV_Name = Generated_files_dir + input_dic['model_name'] + '.csv'	# name of the result file

	data = pd.read_csv("CSV_Results/SouceFile.csv")	# SourceFile.csv is the candidate file list without any class probability. In the next for loop, the class propabilites are are set.
	if not os.path.exists(Generated_files_dir):
		os.makedirs(Generated_files_dir)
	for i in range(len(row_list)):
		data.ix[row_list[i]-2,'probability'] = predict[i]
	data.to_csv(Generated_CSV_Name, sep='\t')



def accuracy_one_fold(input_dic):
	# Accuracy on test fold samples is computed in this function. the result will be saved in a csv file with the same name as the model name.
	data_path = input_dic['data_path'] + str(input_dic['input_shape'][0]) +'x'+ str(input_dic['input_shape'][1]) +'x'+ str(input_dic['input_shape'][2])
	model_path = input_dic['model_path']
	test_fold = input_dic['test_fold']
	input_shape = input_dic['input_shape']

	nodule_dir = data_path+'/Nodule_CandList/'
	non_nodule_dir = data_path + '/Non-Nodule/'
	nodule_fold_dir = nodule_dir + 'Fold' + str(test_fold)	# directory where nodules are stroed
	non_nodule_fold_dir = non_nodule_dir + 'Fold' + str(test_fold)	# directory where negative samples are stored.

	nodule_files = os.listdir(nodule_fold_dir)
	non_nodule_files = os.listdir(non_nodule_fold_dir)

	test_nodule_files_complete_addres = [nodule_fold_dir + '/' + s for s in nodule_files]	
	test_non_nodule_files_complete_addres = [non_nodule_fold_dir + '/' + s for s in non_nodule_files]

	test_files_complete_addres = test_nodule_files_complete_addres + test_non_nodule_files_complete_addres

	test_nodule_labels = [1] * len(test_nodule_files_complete_addres)
	test_non_nodule_labels = [0] * len(test_non_nodule_files_complete_addres)

	test_labels = test_nodule_labels + test_non_nodule_labels
	partition = {'test':test_files_complete_addres}
	labels = dict(zip(test_files_complete_addres, test_labels))


	params = {'dim': input_shape,
			'batch_size': 1,
			'shuffle': False}
	test_generator = DataGenerator(partition['test'], labels, **params)
	model = load_model(model_path + input_dic['model_name'])
	predict = model.predict_generator(test_generator,steps=len(partition['test']),workers=6, verbose=1)

	row_list = [int(x[:-4]) for x in nodule_files + non_nodule_files]	# nodule_files and non_nodule_files lists are lists of positive and negative samples with their name is the same as their row number in candidate list csv file. To find its row in .csv candidate list we just need to remove '.npy' from the end of the filename.
	conf_matrix = confusion_matrix(test_labels, np.argmax(predict,axis=1))
	print(conf_matrix)
	write_csv(input_dic, row_list, predict[:,1])


################# Evaluation Generated Scripts #####################
### The rest is taken from Luna Evaluation Script function #########


# Evaluation settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
FROC_maxX = 8 # Maximum value of x-axis of FROC curve
bLogPlot = True

def generateBootstrapSet(scanToCandidatesDict, FROCImList):
	'''
	Generates bootstrapped version of set
	'''
	imageLen = FROCImList.shape[0]
	
	# get a random list of images using sampling with replacement
	rand_index_im   = np.random.randint(imageLen, size=imageLen)
	FROCImList_rand = FROCImList[rand_index_im]
	
	# get a new list of candidates
	candidatesExists = False
	for im in FROCImList_rand:
		if im not in scanToCandidatesDict:
			continue
		
		if not candidatesExists:
			candidates = np.copy(scanToCandidatesDict[im])
			candidatesExists = True
		else:
			candidates = np.concatenate((candidates,scanToCandidatesDict[im]),axis = 1)

	return candidates

def compute_mean_ci(interp_sens, confidence = 0.95):
	sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
	sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
	sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
	
	Pz = (1.0-confidence)/2.0
		
	for i in range(interp_sens.shape[1]):
		# get sorted vector
		vec = interp_sens[:,i]
		vec.sort()

		sens_mean[i] = np.average(vec)
		# print(Pz)
		# print(len(vec))
		# print(math.floor(Pz*len(vec)))
		# print(vec[math.floor(Pz*len(vec))])
		sens_lb[i] = vec[int(math.floor(Pz*len(vec)))]
		sens_up[i] = vec[int(math.floor((1.0-Pz)*len(vec)))]

	return sens_mean,sens_lb,sens_up

def computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,FROCImList,excludeList,numberOfBootstrapSamples=1000, confidence = 0.95):

	set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)
	
	fps_lists = []
	sens_lists = []
	thresholds_lists = []
	
	FPDivisorList_np = np.asarray(FPDivisorList)
	FROCImList_np = np.asarray(FROCImList)
	
	# Make a dict with all candidates of all scans
	scanToCandidatesDict = {}
	for i in range(len(FPDivisorList_np)):
		seriesuid = FPDivisorList_np[i]
		candidate = set1[:,i:i+1]

		if seriesuid not in scanToCandidatesDict:
			scanToCandidatesDict[seriesuid] = np.copy(candidate)
		else:
			scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid],candidate),axis = 1)

	for i in range(numberOfBootstrapSamples):
		print('computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples))
		# Generate a bootstrapped set
		btpsamp = generateBootstrapSet(scanToCandidatesDict,FROCImList_np)
		fps, sens, thresholds = computeFROC(btpsamp[0,:],btpsamp[1,:],len(FROCImList_np),btpsamp[2,:])
	
		fps_lists.append(fps)
		sens_lists.append(sens)
		thresholds_lists.append(thresholds)

	# compute statistic
	all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
	
	# Then interpolate all FROC curves at this points
	interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
	for i in range(numberOfBootstrapSamples):
		interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
	
	# compute mean and CI
	sens_mean,sens_lb,sens_up = compute_mean_ci(interp_sens, confidence = confidence)

	return all_fps, sens_mean, sens_lb, sens_up

def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
	# Remove excluded candidates
	FROCGTList_local = []
	FROCProbList_local = []
	for i in range(len(excludeList)):
		if excludeList[i] == False:
			FROCGTList_local.append(FROCGTList[i])
			FROCProbList_local.append(FROCProbList[i])
	
	numberOfDetectedLesions = sum(FROCGTList_local)
	totalNumberOfLesions = sum(FROCGTList)
	totalNumberOfCandidates = len(FROCProbList_local)
	fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
	if sum(FROCGTList) == len(FROCGTList): # Handle border case when there are no false positives and ROC analysis give nan values.
	  print("WARNING, this system has no false positives..")
	  fps = np.zeros(len(fpr))
	else:
	  fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
	sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
	return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
				performBootstrapping=False,numberOfBootstrapSamples=1000,confidence = 0.95):
	'''
	function to evaluate a CAD algorithm
	@param seriesUIDs: list of the seriesUIDs of the cases to be processed
	@param results_filename: file with results
	@param outputDir: output directory
	@param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
	@param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
	'''

	nodOutputfile = open(os.path.join(outputDir,'CADAnalysis.txt'),'w')
	nodOutputfile.write("\n")
	nodOutputfile.write((60 * "*") + "\n")
	nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
	nodOutputfile.write((60 * "*") + "\n")
	nodOutputfile.write("\n")

	results = csvTools.readCSV(results_filename)

	allCandsCAD = {}
	
	for seriesuid in seriesUIDs:
		
		# collect candidates from result file
		nodules = {}
		header = results[0]
		header = header[0].split('\t')
		
		i = 0
		for result in results[1:]:
			result = result[0].split('\t')
			nodule_seriesuid = result[header.index(seriesuid_label)]
			
			if seriesuid == nodule_seriesuid:
				nodule = getNodule(result, header)
				nodule.candidateID = i
				nodules[nodule.candidateID] = nodule
				i += 1

		if (maxNumberOfCADMarks > 0):
			# number of CAD marks, only keep must suspicous marks

			if len(nodules.keys()) > maxNumberOfCADMarks:
				# make a list of all probabilities
				probs = []
				for keytemp, noduletemp in nodules.items():
					probs.append(float(noduletemp.CADprobability))
				probs.sort(reverse=True) # sort from large to small
				probThreshold = probs[maxNumberOfCADMarks]
				nodules2 = {}
				nrNodules2 = 0
				for keytemp, noduletemp in nodules.items():
					if nrNodules2 >= maxNumberOfCADMarks:
						break
					if float(noduletemp.CADprobability) > probThreshold:
						nodules2[keytemp] = noduletemp
						nrNodules2 += 1

				nodules = nodules2
		
		print('adding candidates: ' + seriesuid)
		allCandsCAD[seriesuid] = nodules
	
	# open output files
	nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')
	
	# --- iterate over all cases (seriesUIDs) and determine how
	# often a nodule annotation is not covered by a candidate

	# initialize some variables to be used in the loop
	candTPs = 0
	candFPs = 0
	candFNs = 0
	candTNs = 0
	totalNumberOfCands = 0
	totalNumberOfNodules = 0
	doubleCandidatesIgnored = 0
	irrelevantCandidates = 0
	minProbValue = -1000000000.0 # minimum value of a float
	FROCGTList = []
	FROCProbList = []
	FPDivisorList = []
	excludeList = []
	FROCtoNoduleMap = []
	ignoredCADMarksList = []

	# -- loop over the cases
	for seriesuid in seriesUIDs:
		# get the candidates for this case
		try:
			candidates = allCandsCAD[seriesuid]
		except KeyError:
			candidates = {}

		# add to the total number of candidates
		totalNumberOfCands += len(candidates.keys())

		# make a copy in which items will be deleted
		candidates2 = candidates.copy()

		# get the nodule annotations on this case
		try:
			noduleAnnots = allNodules[seriesuid]
		except KeyError:
			noduleAnnots = []

		# - loop over the nodule annotations
		for noduleAnnot in noduleAnnots:
			# increment the number of nodules
			if noduleAnnot.state == "Included":
				totalNumberOfNodules += 1

			x = float(noduleAnnot.coordX)
			y = float(noduleAnnot.coordY)
			z = float(noduleAnnot.coordZ)

			# 2. Check if the nodule annotation is covered by a candidate
			# A nodule is marked as detected when the center of mass of the candidate is within a distance R of
			# the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
			# CT scan, we set R to be the radius of the nodule size.
			diameter = float(noduleAnnot.diameter_mm)
			if diameter < 0.0:
			  diameter = 10.0
			radiusSquared = pow((diameter / 2.0), 2.0)

			found = False
			noduleMatches = []
			for key, candidate in candidates.items():
				x2 = float(candidate.coordX)
				y2 = float(candidate.coordY)
				z2 = float(candidate.coordZ)
				dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
				if dist < radiusSquared:
					if (noduleAnnot.state == "Included"):
						found = True
						noduleMatches.append(candidate)
						if key not in candidates2.keys():
							print("This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (str(candidate.id), seriesuid, str(noduleAnnot.id)))
						else:
							del candidates2[key]
					elif (noduleAnnot.state == "Excluded"): # an excluded nodule
						if bOtherNodulesAsIrrelevant: #    delete marks on excluded nodules so they don't count as false positives
							if key in candidates2.keys():
								irrelevantCandidates += 1
								ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id), float(candidate.CADprobability)))
								del candidates2[key]
			if len(noduleMatches) > 1: # double detection
				doubleCandidatesIgnored += (len(noduleMatches) - 1)
			if noduleAnnot.state == "Included":
				# only include it for FROC analysis if it is included
				# otherwise, the candidate will not be counted as FP, but ignored in the
				# analysis since it has been deleted from the nodules2 vector of candidates
				if found == True:
					# append the sample with the highest probability for the FROC analysis
					maxProb = None
					for idx in range(len(noduleMatches)):
						candidate = noduleMatches[idx]
						if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
							maxProb = float(candidate.CADprobability)

					FROCGTList.append(1.0)
					FROCProbList.append(float(maxProb))
					FPDivisorList.append(seriesuid)
					excludeList.append(False)
					FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
					candTPs += 1
				else:
					candFNs += 1
					# append a positive sample with the lowest probability, such that this is added in the FROC analysis
					FROCGTList.append(1.0)
					FROCProbList.append(minProbValue)
					FPDivisorList.append(seriesuid)
					excludeList.append(True)
					FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), int(-1), "NA"))
					nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(-1)))

		# add all false positives to the vectors
		for key, candidate3 in candidates2.items():
			candFPs += 1
			# print(key)
			# print(candidate3.CADprobability)
			# print(candidate3.seriesuid)
			# print(candidate3.coordX, candidate3.coordY, candidate3.coordZ)
			# raw_input('wait here!')
			FROCGTList.append(0.0)
			FROCProbList.append(float(candidate3.CADprobability))
			FPDivisorList.append(seriesuid)
			excludeList.append(False)
			FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id), float(candidate3.CADprobability)))

	if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
		nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

	nodOutputfile.write("Candidate detection results:\n")
	nodOutputfile.write("    True positives: %d\n" % candTPs)
	nodOutputfile.write("    False positives: %d\n" % candFPs)
	nodOutputfile.write("    False negatives: %d\n" % candFNs)
	nodOutputfile.write("    True negatives: %d\n" % candTNs)
	nodOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)
	nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)

	nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
	nodOutputfile.write("    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)
	if int(totalNumberOfNodules) == 0:
		nodOutputfile.write("    Sensitivity: 0.0\n")
	else:
		nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
	nodOutputfile.write("    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))

	# compute FROC
	fps, sens, thresholds = computeFROC(FROCGTList,FROCProbList,len(seriesUIDs),excludeList)
	
	if performBootstrapping:
		fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up = computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,seriesUIDs,excludeList,
																  numberOfBootstrapSamples=numberOfBootstrapSamples, confidence = confidence)
		
	# Write FROC curve
	with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
		for i in range(len(sens)):
			f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))
	
	# Write FROC vectors to disk as well
	with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
		for i in range(len(FROCGTList)):
			f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

	fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
	
	sens_itp = np.interp(fps_itp, fps, sens)
	
	if performBootstrapping:
		# Write mean, lower, and upper bound curves to disk
		with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
			f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
			for i in range(len(fps_bs_itp)):
				f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
	else:
		fps_bs_itp = None
		sens_bs_mean = None
		sens_bs_lb = None
		sens_bs_up = None

	# create FROC graphs
	if int(totalNumberOfNodules) > 0:
		graphTitle = str("")
		fig1 = plt.figure()
		ax = plt.gca()
		clr = 'b'
		plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
		if performBootstrapping:
			plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
			plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
			plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
			ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
		xmin = FROC_minX
		xmax = FROC_maxX
		plt.xlim(xmin, xmax)
		plt.ylim(0, 1)
		plt.xlabel('Average number of false positives per scan')
		plt.ylabel('Sensitivity')
		plt.legend(loc='lower right')
		plt.title('FROC performance - %s' % (CADSystemName))
		
		if bLogPlot:
			plt.xscale('log', basex=2)
			ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
		
		# set your ticks manually
		ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
		ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
		plt.grid(b=True, which='both')
		plt.tight_layout()

		plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

	return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)
	
def getNodule(annotation, header, state = ""):
	nodule = NoduleFinding()
	nodule.coordX = annotation[header.index(coordX_label)]
	nodule.coordY = annotation[header.index(coordY_label)]
	nodule.coordZ = annotation[header.index(coordZ_label)]
	
	if diameter_mm_label in header:
		nodule.diameter_mm = annotation[header.index(diameter_mm_label)]
	
	if CADProbability_label in header:
		nodule.CADprobability = annotation[header.index(CADProbability_label)]
	
	if not state == "":
		nodule.state = state

	return nodule
	
def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
	allNodules = {}
	noduleCount = 0
	noduleCountTotal = 0
	
	for seriesuid in seriesUIDs:
		print('adding nodule annotations: ' + seriesuid)
		
		nodules = []
		numberOfIncludedNodules = 0
		
		# add included findings
		header = annotations[0]
		for annotation in annotations[1:]:
			nodule_seriesuid = annotation[header.index(seriesuid_label)]
			
			if seriesuid == nodule_seriesuid:
				nodule = getNodule(annotation, header, state = "Included")
				nodules.append(nodule)
				numberOfIncludedNodules += 1
		
		# add excluded findings
		header = annotations_excluded[0]
		for annotation in annotations_excluded[1:]:
			nodule_seriesuid = annotation[header.index(seriesuid_label)]
			
			if seriesuid == nodule_seriesuid:
				nodule = getNodule(annotation, header, state = "Excluded")
				nodules.append(nodule)
			
		allNodules[seriesuid] = nodules
		noduleCount      += numberOfIncludedNodules
		noduleCountTotal += len(nodules)
	
	print('Total number of included nodule annotations: ' + str(noduleCount))
	print('Total number of nodule annotations: ' + str(noduleCountTotal))
	return allNodules
	
	
def collect(annotations_filename,annotations_excluded_filename,seriesuids_filename):
	annotations          = csvTools.readCSV(annotations_filename)
	annotations_excluded = csvTools.readCSV(annotations_excluded_filename)
	seriesUIDs_csv = csvTools.readCSV(seriesuids_filename)
	
	seriesUIDs = []
	for seriesUID in seriesUIDs_csv:
		seriesUIDs.append(seriesUID[0])

	allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)
	
	return (allNodules, seriesUIDs)
	
	
def noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir):
	'''
	function to load annotations and evaluate a CAD algorithm
	@param annotations_filename: list of annotations
	@param annotations_excluded_filename: list of annotations that are excluded from analysis
	@param seriesuids_filename: list of CT images in seriesuids
	@param results_filename: list of CAD marks with probabilities
	@param outputDir: output directory
	'''
	
	print(annotations_filename)
	
	(allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
	
	evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
				os.path.splitext(os.path.basename(results_filename))[0],
				maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
				numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)


def write_excel_files(input_dic):
	# this function calls evaluateCAD function (written as part of luna evaluation script). The inputs of evaluateCAD function is set in the next lines. The output (bootstrap file, FROC curve, CAD analysis, etc. are written in a folder with the same as model name)
	test_fold = input_dic['test_fold']
	model_path = input_dic['model_path']+input_dic['model_name']	# address where the model saved

	annotations_filename = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/LIDC/Mehdi_Summer2018/Luna_Scans/' + 'annotations.csv'	# annotation file address. available to download in luna challenge website
	annotations_excluded_filename = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/LIDC/Mehdi_Summer2018/Luna_Scans/' + 'annotations_excluded.csv'	# excluded file. These are nodules that annotated by 1 or 2 radialogists, nodules < 3 mm in diameter, or non-nodules marked by radiologists. available to download in luna challenge website
	seriesuids_filename = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/LIDC/Mehdi_Summer2018/Luna_Scans/' + 'seriesuids.csv'	# the csv file containing the scan number (available to download in luna challenge website)
	
	Generated_files_dir = model_path + '_csvResults/' 	# the directory where the evaluation files will be generated.

	results_filename = Generated_files_dir + input_dic['model_name'] + '.csv'	# name of the csv result file. same as model name.


	patient_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/LIDC/Mehdi_Summer2018/Luna_Scans/subset' + str(test_fold)	# test scans directory. it is useful to pass the evaluation function just those scans that fall in test fold directory instead of all scans.
	seriesUIDs2 = [f[:-4] for f in os.listdir(patient_dir) if f.endswith('.mhd')]	# scan number is the name of the files in test directory without the extension.
	(allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
	allNodules = {key:value for key, value in allNodules.items() if key in seriesUIDs2}

	evaluateCAD(seriesUIDs2, results_filename, Generated_files_dir, allNodules,
				os.path.splitext(os.path.basename(results_filename))[0],
				maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
				numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)


def compute_CPM(input_dic,history):
	# This function computes the CPM by reading the bootstrapping file generated by evaluation script and interpolating the sensitivities for 8, 4, 2, 1, 0.5, 0.25, and 0.125 FP/scans
	model_path = input_dic['model_path']+input_dic['model_name']	
	Generated_files_dir = model_path + '_csvResults/'	# the directory where the boot strap file was stored.


	froc_address = Generated_files_dir + 'froc_' + input_dic['model_name'] + '_bootstrapping.csv'
	froc = pd.read_csv(froc_address, skipinitialspace=True)	#read boot strap csv file
	FPrate_vect = froc['FPrate'].values	
	sensitivity_vect = froc['Sensivity[Mean]'].values	# sensitivity vector for all the FPs per scans.
	operating_points = np.array([0.125,0.25,0.5,1,2,4,8])	
	CPM_vector = np.interp(operating_points, FPrate_vect, sensitivity_vect)	# interpolating sensitivities for 7 desired operating points.
	print(CPM_vector)
	print(np.mean(CPM_vector))


	train_acc = history['acc']
	train_loss = history['loss']
	val_loss = history['loss']
	val_acc = history['val_acc']	# train/validation loss and accuracy are plotted and the CPM value will be title of this plot.

	fig = plt.figure(1)
	fig.suptitle("CPM: " + "{:.4f}".format(np.mean(CPM_vector)), fontsize="x-large")
	plt.subplot(211)
	plt.plot(range(1,len(history['acc'])+1), history['acc'], 'bo-',label="test1")
	plt.plot(range(1,len(history['acc'])+1), history['val_acc'],'ro-',label="test2")
	plt.grid()

	plt.subplot(212)
	plt.plot(range(1,len(history['acc'])+1), history['loss'], 'bo-')
	plt.plot(range(1,len(history['acc'])+1), history['val_loss'],'ro-')
	plt.title('Loss changes')
	plt.grid()
	plt.savefig(Generated_files_dir + 'history' + '.png')
	plt.cla()




