import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
import time
import cv2
import copy
from scipy.io import savemat

# This script extracts patches, augment nodules and make the files ready for deep learning algorithm. 
# The result patches will be saved at '/raida/shamidian/mehdi/patches/'. This directory can be changed at the 
# end of crop volume section. Other import parameter of the script are defined as follow:




###########################################################################
########################## Extract Patches ################################
###########################################################################

# Patches are extracted around the candidate locations (candidate.csv file) as well as 
# nodule locations (annotation.csv file). Next couple of functions are used to do the job.


def load_itk_image(filename):
	# input: address of .mhd and .raw files
	# output: CT scan stored in numpy array and physical origing and resolution spacing
	itkimage = sitk.ReadImage(filename)
	numpyImage = sitk.GetArrayFromImage(itkimage)
	numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
	numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
	return numpyImage, numpyOrigin, numpySpacing


def readCSV(filename):
	# a routine for reading csv files. return csv file as a 2-D list
	lines = []
	with open(filename, "rb") as f:
		csvreader = csv.reader(f)
		for line in csvreader:
			lines.append(line)
	return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
	# convert world coordinate to voxel coordinates
	stretchedVoxelCoord = np.absolute(worldCoord - origin)
	voxelCoord = stretchedVoxelCoord / spacing
	return voxelCoord.astype(int)

def normalizePlanes(npzarray):
	# Clip HUs to [-1000,400], and normalize it to [0,1]
	maxHU = 400.
	minHU = -1000.
	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray>1] = 1.
	npzarray[npzarray<0] = 0.
	return npzarray

def uniform_res(volume,orign_res, dest_res):
	# input: 3-D scan as the numpy array, origin_res: resolution of the scan, dest_res: final resolution we want to reach.
	xy_res = orign_res[2]
	z_res = orign_res[0]
	resize_factor = [xy_res/dest_res[0],xy_res/dest_res[0],z_res/dest_res[2]]  # based on the origin_res and dest_res we compute the resize factor
	niformed_volume = ndimage.interpolation.zoom(volume,zoom = tuple(resize_factor),order = 3) # resize the image to have the desired resolution
	return niformed_volume, resize_factor  #resize factor is returned back as well because it will be used to update the coordinate of nodule location.

def extract_nodules(load_dir_prefix, save_dir_prefix,patch_sizeH):
	# The function extracts the nodules by reading the annotation csv files
	# in this function both the ct scans and nodule coordinates are revised to match
	# the destination resolution. The patches are extracted from CT scans (after they went through re-sampling)
	# and are saved into save_dir.
	annotation_file = load_dir_prefix + '/annotations.csv'
	annotation_list = readCSV(annotation_file)
	seriesuidS = [i[0] for i in annotation_list]

	for i in range(10):
		print(i)
		fold = load_dir_prefix + '/subset' + str(i)
		for name in glob.glob(fold+'/*.mhd'):
			seriesuid = name[-68:-4] # seriesuid is the same as the file name (excluding the extension of the file name)
			ct, numpyOrigin, numpySpacing = load_itk_image(name)
			ct = ct.astype(np.float32)

			ct_new = np.copy(ct)
			tmp = np.swapaxes(ct_new,0,2)
			ct_new = np.swapaxes(tmp,0,1)  # axis swapped in order to have the axial view of the scan

			ct_new, resize_factor = uniform_res(ct_new, numpySpacing,dest_res) # resolution of the scan uniformed.
			ct_new = normalizePlanes(ct_new) # convert the HUs to the range of (0,1)
			ct_new = np.pad(ct_new,((patch_sizeH[0],patch_sizeH[0]),(patch_sizeH[1],patch_sizeH[1]),(patch_sizeH[2],patch_sizeH[2])),'constant',constant_values = 0) # padding the scan with half the size the batch size because the nodule coordinate might be in the border of scan.

			matches = [y for y, x in enumerate(seriesuidS) if x == seriesuid]  #find the rows of the csv file that correspond to the same patient (seriesuid)

			for match in matches:
				print(match)
				worldCoord = np.asarray([float(annotation_list[match][3]),float(annotation_list[match][2]),float(annotation_list[match][1])]) # read the coordinate of the nodule in csv file
				voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing) # convert world coordinate to voxel coordinate
				
				voxelCoord_new = np.copy(voxelCoord)
				voxelCoord_new[0], voxelCoord_new[2] = voxelCoord_new[2], voxelCoord_new[0]
				voxelCoord_new[0], voxelCoord_new[1] = voxelCoord_new[1], voxelCoord_new[0] # the x, y, and z coordinates are swapped the same way that ct scan coordinates swapped earlier.

				voxelCoord_new = np.round(voxelCoord_new*np.asarray(resize_factor)).astype(int) # Since the CT scan resized we have to update the nodule coordinate
				voxelCoord_new = voxelCoord_new + patch_sizeH # because of the padding to the CT scan earlier, need to add the same padding size to the coordinate.

				patch_new = ct_new[voxelCoord_new[0]-patch_sizeH[0]:voxelCoord_new[0]+patch_sizeH[0]+1,\
							voxelCoord_new[1]-patch_sizeH[1]:voxelCoord_new[1]+patch_sizeH[1]+1,\
							voxelCoord_new[2]-patch_sizeH[2]:voxelCoord_new[2]+patch_sizeH[2]+1] # extract the patch from the center of the nodule coordinate


				save_dir = os.path.join(save_dir_prefix, str(patch_sizeB[0])+'x'+str(patch_sizeB[1])+'x'+str(patch_sizeB[2]), 'Nodule','Fold'+str(i))
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)
				np.save(save_dir + str(match+1),patch_new)  # The name of the patch - that is saved to the hard disk - is the row number in annotation csv file.


def extract_candidates(dir_prefix,patch_sizeB):
	# The function is the same as the extract nodules function, but instead of nodules it extracts the candidates.
	# for further detail about each each line, go to the corresponding line in extract_nodules function
	# This function may be merged with extract_nodules function in the future. 
	load_dir_prefix = dir_prefix
	patch_sizeH = [i/2 for i in patch_sizeB]

	patch_sizeH = [i/2 for i in patch_size]
	candidate_file = load_dir_prefix + '/candidates_V2.csv'
	candidate_list = readCSV(candidate_file)
	lesion_type = [i[4] for i in candidate_list]
	seriesuidS = [i[0] for i in candidate_list]

	for i in range(10):
		print(i)
		fold = load_dir_prefix + '/subset' + str(i)
		for name in glob.glob(fold+'/*.mhd'):
			seriesuid = name[-68:-4]
			ct, numpyOrigin, numpySpacing = load_itk_image(name)
			ct = ct.astype(np.float32)

			ct_new = np.copy(ct)
			tmp = np.swapaxes(ct_new,0,2)
			ct_new = np.swapaxes(tmp,0,1)

			ct_new = np.pad(ct_new,((patch_sizeH[0],patch_sizeH[0]),(patch_sizeH[1],patch_sizeH[1]),(patch_sizeH[2],patch_sizeH[2])),'constant',constant_values = 0)

			matches = [y for y, x in enumerate(seriesuidS) if x == seriesuid]

			for match in matches:

				worldCoord = np.asarray([float(candidate_list[match][3]),float(candidate_list[match][2]),float(candidate_list[match][1])])
				voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
				
				voxelCoord_new = np.copy(voxelCoord)
				voxelCoord_new[0], voxelCoord_new[2] = voxelCoord_new[2], voxelCoord_new[0]
				voxelCoord_new[0], voxelCoord_new[1] = voxelCoord_new[1], voxelCoord_new[0]

				voxelCoord_new = voxelCoord_new + patch_sizeH

				patch_new = ct_new[voxelCoord_new[0]-patch_sizeH[0]:voxelCoord_new[0]+patch_sizeH[0]+1,\
									voxelCoord_new[1]-patch_sizeH[1]:voxelCoord_new[1]+patch_sizeH[1]+1,\
									voxelCoord_new[2]-patch_sizeH[2]:voxelCoord_new[2]+patch_sizeH[2]+1]
				
				if(lesion_type[match] == '0'):  # if non-nodule
					save_dir = os.path.join(load_dir_prefix, 'patch', str(patch_size[0])+'x'+str(patch_size[1])+'x'+ str(patch_size[2]), 'Non-Nodule', 'Fold'+str(i))
				if(lesion_type[match] == '1'):  # if non-nodule
					save_dir = os.path.join(load_dir_prefix, 'patch', str(patch_size[0])+'x'+str(patch_size[1])+'x'+ str(patch_size[2]), 'Nodule', 'Fold'+str(i))
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)
				np.save(save_dir + str(match+1),patch_new)

	

################################################################################
################################ AUGMENTATION ##################################
################################################################################



def rotate_image(image, angle):
		"""
		Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
		(in degrees). The returned image will be large enough to hold the entire
		new image, with a black background
		"""

		# Get the image size
		# No that's not an error - NumPy stores image matricies backwards
		image_size = (image.shape[1], image.shape[0])
		image_center = tuple(np.array(image_size) / 2)

		# Convert the OpenCV 3x2 rotation matrix to 3x3
		rot_mat = np.vstack(
				[cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
		)

		rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

		# Shorthand for below calcs
		image_w2 = image_size[0] * 0.5
		image_h2 = image_size[1] * 0.5

		# Obtain the rotated coordinates of the image corners
		rotated_coords = [
				(np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
				(np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
				(np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
				(np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
		]

		# Find the size of the new image
		x_coords = [pt[0] for pt in rotated_coords]
		x_pos = [x for x in x_coords if x > 0]
		x_neg = [x for x in x_coords if x < 0]

		y_coords = [pt[1] for pt in rotated_coords]
		y_pos = [y for y in y_coords if y > 0]
		y_neg = [y for y in y_coords if y < 0]

		right_bound = max(x_pos)
		left_bound = min(x_neg)
		top_bound = max(y_pos)
		bot_bound = min(y_neg)

		new_w = int(abs(right_bound - left_bound))
		new_h = int(abs(top_bound - bot_bound))

		# We require a translation matrix to keep the image centred
		trans_mat = np.matrix([
				[1, 0, int(new_w * 0.5 - image_w2)],
				[0, 1, int(new_h * 0.5 - image_h2)],
				[0, 0, 1]
		])

		# Compute the tranform for the combined rotation and translation
		affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

		# Apply the transform
		result = cv2.warpAffine(
				image,
				affine_mat,
				(new_w, new_h),
				flags=cv2.INTER_LINEAR
		)

		return result

def augment_nodules(dir_prefix, save_dir):
	# the function augments all the nodules found in directory load_dir
	# and save the augmented samples in the save_dir directory.
	load_dir = dir_prefix + 'patches/'
	image_files = os.listdir(load_dir)
	num_images = len(image_files)
	plt.ion()
	for img_name in image_files:
			print(img_name)
			image_address = load_dir + '/' + img_name
			img = np.load(image_address)
			image_height, image_width = img.shape[0:2]

			# save patch itself
			dest_address = save_dir + img_name[:-4] + '_Patch' + '{0:03d}'.format(0)
			np.save(dest_address,img)   

			# Flip in the direction of x coordinate:
			img2 = np.zeros(shape = img.shape, dtype = img.dtype)
			img2 = np.flip(img,0)
			dest_address = save_dir + img_name[:-4] + '_FlipX' + '{0:03d}'.format(0)
			np.save(dest_address,img2)

			# Flip in the direction of y coordinate:
			img2 = np.zeros(shape = img.shape, dtype = img.dtype)
			img2 = np.flip(img,1)
			dest_address = save_dir + img_name[:-4] + '_FlipY' + '{0:03d}'.format(0)
			np.save(dest_address,img2)

			# Flip in the direction of z coordinate:
			img2 = np.zeros(shape = img.shape, dtype = img.dtype)
			img2 = np.flip(img,2)
			dest_address = save_dir + img_name[:-4] + '_FlipZ' + '{0:03d}'.format(0)
			np.save(dest_address,img2)

			# shift left and right in x direction for values (1,2,3)
			for i in range(1,4,1):
				img2 = np.zeros(shape = img.shape, dtype = img.dtype)
				img2 = np.roll(img,i,axis=0)  # shift right
				dest_address = save_dir + img_name[:-4] + '_ShftX' + '{0:03d}'.format(i)
				np.save(dest_address,img2)
				img2 = np.roll(img,-i,axis=0) # shift left
				dest_address = save_dir + img_name[:-4] + '_ShftX' + '{0:03d}'.format(-i)
				np.save(dest_address,img2)

			# shift up and down in y direction for values (1,2,3)
			for i in range(1,4,1):
				img2 = np.zeros(shape = img.shape, dtype = img.dtype)
				img2 = np.roll(img,i,axis=1)  # shift up
				dest_address = save_dir + img_name[:-4] + '_ShftY' + '{0:03d}'.format(i)
				np.save(dest_address,img2)
				img2 = np.roll(img,-i,axis=1) # shift down
				dest_address = save_dir + img_name[:-4] + '_ShftY' + '{0:03d}'.format(-i)
				np.save(dest_address,img2)

			# shift in z direction for values (1,2,3)
			for i in range(1,4,1):
				img2 = np.zeros(shape = img.shape, dtype = img.dtype)
				img2 = np.roll(img,i,axis=(0,1))
				dest_address = save_dir + img_name[:-4] + '_ShftD' + '{0:03d}'.format(i)
				np.save(dest_address,img2)
				img2 = np.roll(img,-i,axis=(0,1))
				dest_address = save_dir + img_name[:-4] + '_ShftD' + '{0:03d}'.format(-i)
				np.save(dest_address,img2)
			

			# Rotate in x-y plane 
			for i in np.arange(10,359,10):
				img2 = np.zeros(shape = img.shape, dtype = img.dtype)
				for j in range(img2.shape[2]):
						plane = rotate_image(img[:,:,j],i)

						plane = plane[int((plane.shape[0]-image_height)/2):int((plane.shape[0]+image_height)/2),int((plane.shape[1]-image_width)/2):int((plane.shape[1]+image_width)/2)]
						img2[:,:,j] = plane
				dest_address = save_dir + img_name[:-4] + '_Rotat' + '{0:03d}'.format(i) 
				np.save(dest_address,img2)


def augment_all_folds(load_dir_prefix, patch_sizeB):
	# This function just go through all folds and call augment_nodules function for each fold
	for i in range(10):
		nodule_directory = os.path.join(load_dir_prefix , str(patch_sizeB[0]) + 'x' + str(patch_sizeB[1]) + 'x' + str(patch_sizeB[2]) , 'Nodule', 'Fold' +  str(i))
		augmented_directory = os.path.join(load_dir_prefix, str(patch_sizeB[0]) + 'x' + str(patch_sizeB[1]) + 'x' + str(patch_sizeB[2]), 'Nodule_Augmented', 'Fold'+str(i))
		if not os.path.exists(augmented_directory):
			os.makedirs(augmented_directory)
		augment_nodules(nodule_directory,augmented_directory)


augment_all_folds(load_dir_prefix, patch_sizeB)



#####################################################
############# INSPECT FILES IN A FOLDER #############
#####################################################

class IndexTracker(object):
		def __init__(self, ax, X):
				self.ax = ax
				ax.set_title('use scroll wheel to navigate images')

				self.X = X
				rows, cols, self.slices = X.shape
				self.ind = self.slices//2

				# self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin = 0, vmax = 1)
				self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
				self.update()

		def onscroll(self, event):
				print("%s %s" % (event.button, event.step))
				if event.button == 'up':
						self.ind = (self.ind + 1) % self.slices
				else:
						self.ind = (self.ind - 1) % self.slices
				self.update()

		def update(self):
				self.im.set_data(self.X[:, :, self.ind])
				ax.set_ylabel('slice %s' % self.ind)
				self.im.axes.figure.canvas.draw()

def visualization():
	# visualize the nodules/candidates in a directory saved as numpy files
	files_dir = '/gpfs_projects/mohammadmeh.farhangi/shamidian/mehdi/patches/91x91x25/Nodule/Fold1/' # A directory in which the files will be inspected.

	files = os.listdir(files_dir)
	files.sort()
	for i in range(len(files)):
		print(files[i])
		# if files[i] != '319.npy':
		# 	continue
		patch = np.load(files_dir + files[i])


		fig, ax = plt.subplots(1, 1)
		tracker = IndexTracker(ax, patch)
		fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
		plt.show()



	input('Visualizaton in a folder is finished ...')


if __name__ == "__main__":
	load_dir_prefix = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/LIDC/Mehdi_Summer2018/Luna_Scans/'   # directory where Luna scans are located.
	patch_size = [91,91,25] # patch size to extract negative samples, since negative samples are not subjected to augmentation (e.g. 45 degrees rotation)
	dest_res = [0.625,0.625,2]   # uniform resolution along each axis
	extract_candidates(load_dir_prefix,patch_size)
	raw_input('Patches are extracted')
	augment_all_folds(load_dir_prefix, patch_size)
	raw_input('Nodules were augmented')
	visualization()




