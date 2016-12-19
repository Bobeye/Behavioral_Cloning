import pandas as pd
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split

data_path = '/home/bowen/Downloads/udacity-master/P2/'
img_path = 'IMG/'
log_path = 'driving_log.csv'


def dataset_for_train():
	df = pd.read_csv(data_path+log_path, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
	cam_center = list(df['center'])[1:]
	cam_left = list(df['left'])[1:]
	cam_right = list(df['right'])[1:]
	car_steering = np.array(list(df['steering'])[1:])

	nb_labels = len(car_steering)
	imgs = []
	labels = []

	for i in range(nb_labels):
		label = float(car_steering[i])
		# load images from center, right, left cameras
		Cimg = cv2.imread(data_path+img_path+cam_center[i].split('/')[-1],cv2.IMREAD_COLOR)
		Rimg = cv2.imread(data_path+img_path+cam_right[i].split('/')[-1],cv2.IMREAD_COLOR)
		Limg = cv2.imread(data_path+img_path+cam_left[i].split('/')[-1],cv2.IMREAD_COLOR)
		# normalization
		Rimg = cv2.normalize(Rimg, (0,255), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		Cimg = cv2.normalize(Cimg, (0,255), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		Limg = cv2.normalize(Limg, (0,255), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		# crop images to remove sky, trees, etc.
		Ccrop = Cimg[60:140, 0:320]
		Rcrop = Rimg[60:130, 0:320]
		Lcrop = Limg[60:130, 0:320]
		# transfer to YUV space
		Cyuv = cv2.cvtColor(Ccrop, cv2.COLOR_BGR2YUV)
		Ryuv = cv2.cvtColor(Lcrop, cv2.COLOR_BGR2YCrCb)
		Lyuv = cv2.cvtColor(Rcrop, cv2.COLOR_BGR2YCrCb)
		# resize to 64x16
		imgs += [cv2.resize(Cyuv, (64,16))]
		labels += [label]
		imgs += [cv2.resize(Ryuv, (64,16))]
		labels += [label+0.2]
		imgs += [cv2.resize(Lyuv, (64,16))]
		labels += [label-0.2]

	imgs = np.array(imgs)
	labels = np.array(labels)
	imgs = imgs.reshape((imgs.shape[0], 64, 16, 3))

	# devide into train, val, test dataset
	X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.1, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

	print('train size: ', X_train.shape[0])
	print('valid size: ', X_val.shape[0])
	print('test size: ', X_test.shape[0])

	return (X_train, y_train), (X_val, y_val), (X_test, y_test)
