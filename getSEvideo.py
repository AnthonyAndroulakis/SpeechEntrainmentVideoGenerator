import os
import sys
import glob

import numpy as np
import dlib
import cv2

from video_stream import VideoStream

FACE_DETECTOR_MODEL = None
LANDMARKS_PREDICTOR = None

IMAGE_WIDTH = 500 # Every frame will be resized to this width before any processing

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
	for i in range(48, 68):
		point = landmarks.part(i)
		coords.append((point.x, point.y))

	return np.array(coords)

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

def crop_and_store(frame, mouth_coordinates, name):
	"""
	Args:
		1. frame:				The frame which has to be cropped.
		2. mouth_coordinates:	The coordinates which help in deciding which region is to be cropped.
		3. name:				The path name to be used for storing the cropped image.
	"""

	# Find bounding rectangle for mouth coordinates
	x, y, w, h = cv2.boundingRect(mouth_coordinates)

	mouth_roi = frame[y:y + h, x:x + w]

	h, w, channels = mouth_roi.shape
	# If the cropped region is very small, ignore this case.
	if h < 10 or w < 10:
		return
	
	resized = resize(mouth_roi, 32, 32)
	cv2.imwrite(name, resized)

def extract_mouth_regions(path, output_dir, screen_display):
	"""
	Args:
		1. path:			File path of the video file (.mp4 file) from which lip regions will be cropped.
		2. output_dir:		The dir path where the 32*32 sized mouth regions will be stored.
		3. screen_display:	Decides whether to use screen (to display video being processed).
	"""
	video_name = path.split('/')[-1].split(".")[0]

	stream = VideoStream(path)
	stream.start()

	count = 0 # Counts number of mouth regions extracted

	while not stream.is_empty():
		frame = stream.read()

		frame = resize(frame, IMAGE_WIDTH)

		rects = FACE_DETECTOR_MODEL(frame, 0)

		all_mouth_coordinates = [] 
		# Keeps hold of all mouth coordinates found in the frame.

		for rect in rects:
			landmarks = LANDMARKS_PREDICTOR(frame, rect)
			mouth_coordinates = get_mouth_coord(landmarks)
			all_mouth_coordinates.append(mouth_coordinates)

			crop_and_store(
				frame, 
				mouth_coordinates, 
				name = output_dir +'/'+ str(count) + '.jpg')

			count+=1

		if screen_display:
			visualize(frame, all_mouth_coordinates)			

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	stream.stop()
	if screen_display:
		cv2.destroyAllWindows()


def getSEvideo(path, outputdirectory): #input video path, output video directory
	"""
	Args:
		1. video_dir:			Directory storing all videos to be processed.
	"""

	output_dir = outputdirectory

	os.system('mkdir -p audio')
	os.system('mkdir -p pictures')
	os.system('mkdir -p silentvid')

	originalpath = path

	video_name = path.split('/')[-1].split(".")[0]
	input_directory = '/'.join(path.split('/')[:-1])
	if path[-3:] != 'mp4': 
		os.system('ffmpeg -i '+path+' '+input_directory+'/'+video_name+'.mp4') #convert to .mp4 if not 
		path = input_directory+'/'+video_name+'.mp4' #redefine path

	load_trained_models()

	if not FACE_DETECTOR_MODEL:
		return False

	os.system('ffmpeg -i '+path+' -vn -acodec pcm_s16le -ar 44100 -ac 2 audio/'+video_name+'.wav') #extract audio from video, place in audio folder

	extract_mouth_regions(path, 'pictures', screen_display=False) #pictures placed in pictures folder

	#create SE video
	framerate = os.popen('ffmpeg -i '+path+' 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"').read()[:-1] #as a string
	os.system('ffmpeg -framerate '+framerate+' -start_number 0 -i pictures/%d.jpg -vcodec mpeg4 silentvid/'+video_name+'_silent.mp4') #silent SE video placed in 'silentvid folder

	#add audio to silent video
	os.system('ffmpeg -i silentvid/'+video_name+'_silent.mp4 -i audio/'+video_name+'.wav -c:v copy -c:a aac -strict experimental '+output_dir+'/'+video_name+'SE.mp4')

	os.system('rm -rf audio')
	os.system('rm -rf pictures')
	os.system('rm -rf silentvid')

	if originalpath != path:
		os.system('rm '+path)

	return True

if len(sys.argv[1:])<2:
	print("**Error: no inputs. Correct usage for getSEvideo shown below:")
	print("python3 getSEvideo.py 'input video' 'output folder'")
	print("output video will be in output folder and be named similarly to the input video")
	exit()

getSEvideo(sys.argv[1],sys.argv[2])
#dont bother automating the video recording, just do it on a phone