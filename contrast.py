import os
from PIL import Image, ImageEnhance
import glob
import numpy as np
import cv2


if(not os.path.isdir('contrast')):
	os.makedirs('contrast')
	for dir in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']:
		os.makedirs('contrast/' + dir)

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
		print('Contrast processing (1.3) : ' + imgFileDir )
		imgFile = Image.open(imgFileDir)
		output = ImageEnhance.Contrast(imgFile).enhance(1.3)
		output.save( imgFileDir.replace('train','contrast') )



