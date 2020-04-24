import os
from PIL import Image
import glob
import numpy as np
import cv2

if(not os.path.isdir('edgeEnhance')):
	os.makedirs('edgeEnhance')
	for dir in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']:
		os.makedirs('edgeEnhance/' + dir)

akiec_filenames = glob.glob('train/akiec/*.jpg')
bcc_filenames = glob.glob('train/bcc/*.jpg')
bkl_filenames = glob.glob('train/bkl/*.jpg')
df_filenames = glob.glob('train/df/*.jpg')
mel_filenames = glob.glob('train/mel/*.jpg')
nv_filenames = glob.glob('train/nv/*.jpg')
vasc_filenames = glob.glob('train/vasc/*.jpg')

for classList in [akiec_filenames, bcc_filenames, bkl_filenames, 
			df_filenames, mel_filenames, nv_filenames, vasc_filenames]:

	for imgFileDir in classList:
		print('Edge Enhance processing : ' + imgFileDir )
		imgFile = cv2.imread(imgFileDir)
		#generating the kernels
		kernel = np.array([[-1,-1,-1,-1,-1],
					[-1,2,2,2,-1],
					[-1,2,8,2,-1],
					[-1,-1,-1,-1,-1]])/8.0
		output = cv2.filter2D(imgFile, -1, kernel)
		cv2.imwrite(imgFileDir.replace('train','edgeEnhance'),output)



