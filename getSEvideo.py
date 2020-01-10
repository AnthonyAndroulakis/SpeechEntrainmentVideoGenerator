import os,sys,glob

import numpy as np
import dlib
import cv2
from pykalman import KalmanFilter

from PIL import Image

FACE_DETECTOR_MODEL = None
LANDMARKS_PREDICTOR = None

def kalman_filter(measurements): #thank you kabdulla at https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data
	initial_state_mean = [measurements[0, 0],0,measurements[0, 1],0]
	transition_matrix = [[1, 1, 0, 0],[0, 1, 0, 0],[0, 0, 1, 1],[0, 0, 0, 1]]
	observation_matrix = [[1, 0, 0, 0],[0, 0, 1, 0]]
	kf1 = KalmanFilter(transition_matrices = transition_matrix, observation_matrices = observation_matrix, initial_state_mean = initial_state_mean)
	kf1 = kf1.em(measurements, n_iter=5)
	kf2 = KalmanFilter(transition_matrices = transition_matrix, observation_matrices = observation_matrix, initial_state_mean = initial_state_mean, observation_covariance = 10*kf1.observation_covariance, em_vars=['transition_covariance', 'initial_state_covariance'])
	kf2 = kf2.em(measurements, n_iter=5)
	smoothed_state_means, smoothed_state_covariances  = kf2.smooth(measurements)
	return smoothed_state_means[:, 0], smoothed_state_means[:, 2]

def load_trained_models():
	"""
		Helper function to load DLIB's models.
	"""
	if not os.path.isfile("./model/shape_predictor_68_face_landmarks.dat"):
		return
	global FACE_DETECTOR_MODEL, LANDMARKS_PREDICTOR

	FACE_DETECTOR_MODEL = dlib.get_frontal_face_detector()
	LANDMARKS_PREDICTOR = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

def resize(image, width, height=None):
	"""
	Args:
		1. image: 	Image which has to be resized.
		2. width: 	New width for the image.
		3. height:	New height for the image.
	"""
	h,w,channels = image.shape
	if not height:
		ratio = float(width) / w
		height = int(ratio*h)

	resized = cv2.resize(image, (width, height))
	return resized

def get_mouth_coord(landmarks):
	"""
		Returns mouth region's landmarks in numpy array.
	Args:
		1. landmarks: 	Facial landmarks returned by DLIB's LANDMARKS_PREDICTOR
	"""
	coords = []
	coords.append((landmarks.part(33).x, landmarks.part(33).y))
	coords.append(((landmarks.part(5).x+landmarks.part(6).x)/2, (landmarks.part(5).y+landmarks.part(6).y)/2))
	coords.append(((landmarks.part(10).x+landmarks.part(11).x)/2, (landmarks.part(10).y+landmarks.part(11).y)/2))

	return np.array(coords, dtype=np.float32)

def visualize(frame, coordinates_list, alpha = 0.80, color=[255, 255, 255]):
	"""
	Args:
		1. frame:				OpenCV's image which has to be visualized.
		2. coordinates_list:	List of coordinates which will be visualized in the given `frame`
		3. alpha, color:		Some parameters which help in visualizing properly. 
								A convex hull will be shown for each element in the `coordinates_list` 
	"""
	layer = frame.copy()
	output = frame.copy()

	for coordinates in coordinates_list:
		c_hull = cv2.convexHull(coordinates)
		cv2.drawContours(layer, [c_hull], -1, color, -1)

	cv2.addWeighted(layer, alpha, output, 1 - alpha, 0, output)
	cv2.imshow("Output", output)

def crop_and_store(frame, mouth_coordinates, name, thewidth, theheight):
	"""
	Args:
		1. frame:				The frame which has to be cropped.
		2. mouth_coordinates:	The coordinates which help in deciding which region is to be cropped.
		3. name:				The path name to be used for storing the cropped image.
	"""

	# Find bounding rectangle for mouth coordinates
	x, y, w, h = cv2.boundingRect(mouth_coordinates) #############

	mouth_roi = frame[y:y + h, x:x + w]

	h, w, channels = mouth_roi.shape
	# If the cropped region is very small, ignore this case.
	if h < 10 or w < 10:
		return
	
	resized = resize(mouth_roi, thewidth, theheight)
	cv2.imwrite(name, resized)

def extract_mouth_regions(path, output_dir):
	"""
	Args:
		1. path:			File path of the video file (.mp4 file) from which lip regions will be cropped.
		2. output_dir:		The dir path where the 32*32 sized mouth regions will be stored.
		3. screen_display:	Decides whether to use screen (to display video being processed).
	"""
	video_name = path.split('/')[-1].split(".")[0]

	count = 0 # Counts number of mouth regions extracted

	widths = []
	heights = []
	nose = []
	lchin = []
	rchin = []
	for i in range(len(glob.glob('./frames/*.jpg'))): #find average width and height
		frame = cv2.imread('./frames/'+str(i+1)+'.jpg')
		rects = FACE_DETECTOR_MODEL(frame, 0)
		for rect in rects: #find avg width and height
			landmarks = LANDMARKS_PREDICTOR(frame, rects[0])
			mouth_coordinates = get_mouth_coord(landmarks)
			nose.append((mouth_coordinates[0][0],mouth_coordinates[0][1]))
			lchin.append((mouth_coordinates[1][0],mouth_coordinates[1][1]))
			rchin.append((mouth_coordinates[2][0],mouth_coordinates[2][1]))
			x, y, w, h = cv2.boundingRect(mouth_coordinates)
			widths.append(w)
			heights.append(h)
	avgwidth = round(sum(widths)/len(widths))
	avgheight = round(sum(heights)/len(heights))
	x1,y1 = kalman_filter(np.array(nose))
	nose_pts = np.array((x1, y1)).T
	x2,y2 = kalman_filter(np.array(lchin))
	lchin_pts = np.array((x2, y2)).T
	x3,y3 = kalman_filter(np.array(rchin))
	rchin_pts = np.array((x3, y3)).T

	for i in range(len(glob.glob('./frames/*.jpg'))):
		#frame = stream.read()
		frame = cv2.imread('./frames/'+str(i+1)+'.jpg')

		rects = FACE_DETECTOR_MODEL(frame, 0)

		all_mouth_coordinates = [] 
		# Keeps hold of all mouth coordinates found in the frame.

		for rect in rects:
			mouth_coordinates = [] #reset mouth_coordinates
			mouth_coordinates.append((nose_pts[i][0],nose_pts[i][1])) #i is the frame num
			mouth_coordinates.append((lchin_pts[i][0],lchin_pts[i][1]))
			mouth_coordinates.append((rchin_pts[i][0],rchin_pts[i][1]))
			mouth_coordinates = np.array(mouth_coordinates, dtype=np.float32)
			all_mouth_coordinates.append(mouth_coordinates)

			crop_and_store(
				frame, 
				mouth_coordinates, 
				name = output_dir +'/'+ str(count) + '.jpg',
				thewidth = avgwidth,
				theheight = avgheight)

			count+=1

def getSEvideo(path, outputpath): #input video relative path, output video relative path, number of smoothings by interpolation
	originalpath = path

	os.system('mkdir -p frames')
	os.system('mkdir -p audio')
	os.system('mkdir -p pictures')
	os.system('mkdir -p silentvid')
	os.system('mkdir -p pngfiles')

	print('Preprocessing video...')
	video_name = path.split('/')[-1].split(".")[0]
	input_directory = '/'.join(path.split('/')[:-1])
	if path[-3:] != 'mp4': 
		os.system('ffmpeg -nostats -loglevel 0 -i '+path+' -q:v 0 '+input_directory+'/'+video_name+'.mp4')
		path = input_directory+'/'+video_name+'.mp4'

	#output_dir = outputdirectory
	outvideo_name = outputpath.split('/')[-1].split(".")[0]
	output_dir = '/'.join(outputpath.split('/')[:-1])

	print('Loading face detector model...')
	load_trained_models()

	if not FACE_DETECTOR_MODEL:
		return False

	print('Extracting frames...')
	os.system('ffmpeg -nostats -loglevel 0 -i '+originalpath+' -qscale 0 frames/%d.jpg') #convert video to frames

	print('Extracting audio...')
	os.system('ffmpeg -nostats -loglevel 0 -i '+path+' -qscale 0 audio/'+video_name+'.wav') #extract audio from video, place in audio folder...seems like qscale is fixing everything for me

	print('Extracting mouth regions...')
	extract_mouth_regions(path, './pictures') #pictures placed in pictures folder

	#get png files
	print('Converting images...')
	for k in glob.glob('pictures/*.jpg'):
		jpgim = Image.open(k)
		jpgim.save('pngfiles/'+k.split('/')[-1][:-4]+'.png')

	#create SE video
	print('Creating Speech Entrainment video...')
	framerate = float(os.popen('ffmpeg -i '+path+' 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"').read()[:-1]) #as a float
	os.system('ffmpeg -nostats -loglevel 0 -framerate '+str(framerate)+' -start_number 0 -i pngfiles/%d.png -qscale 0 silentvid/'+video_name+'_silent.mp4') #silent SE video placed in 'silentvid folder

	#add audio to silent video
	print('Adding audio to Speech Entrainment video...')
	os.system('ffmpeg -nostats -loglevel 0 -i silentvid/'+video_name+'_silent.mp4 -i audio/'+video_name+'.wav -c:v copy -c:a aac -strict experimental '+output_dir+'/'+outvideo_name+'_long.mp4') #only outputs to .mp4, you can change the extension afterwards if necessary
	os.system('ffmpeg -nostats -loglevel 0 -i '+output_dir+'/'+outvideo_name+'_long.mp4 -i '+output_dir+'/'+outvideo_name+'_long.mp4 -c copy -shortest -map 0:v -map 1:a '+output_dir+'/'+outvideo_name+'.mp4')

	os.system('rm -rf frames')
	os.system('rm -rf audio')
	os.system('rm -rf pictures')
	os.system('rm -rf silentvid')
	os.system('rm -rf pngfiles')
	if originalpath != path:
		os.system('rm '+path)
	os.system('rm '+output_dir+'/'+outvideo_name+'_long.mp4')

	print('Speech Entrainment video created!')

	return True

if len(sys.argv[1:])<2:
	print("**Error: not enough inputs. Correct usage for getSEvideo shown below:")
	print("python3 getSEvideo.py 'input video relative-path-to-getSEvideo.py' 'output video relative-path-to-getSEvideo.py'")
	exit()

getSEvideo(sys.argv[1],sys.argv[2])
#dont bother automating the video recording, just do it on a phone
