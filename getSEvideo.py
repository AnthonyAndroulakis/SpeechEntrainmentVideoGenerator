import os,sys,glob

import numpy as np
import dlib
import cv2

from PIL import Image

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
	#for i in range(48, 68):
	#	point = landmarks.part(i)
	#	coords.append((point.x, point.y))
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
	#print(mouth_coordinates)
	x, y, w, h = cv2.boundingRect(mouth_coordinates) #############

	mouth_roi = frame[y:y + h, x:x + w]

	h, w, channels = mouth_roi.shape
	# If the cropped region is very small, ignore this case.
	if h < 10 or w < 10:
		return
	
	resized = resize(mouth_roi, thewidth, theheight)
	cv2.imwrite(name, resized)

def extract_mouth_regions(path, output_dir, screen_display):
	"""
	Args:
		1. path:			File path of the video file (.mp4 file) from which lip regions will be cropped.
		2. output_dir:		The dir path where the 32*32 sized mouth regions will be stored.
		3. screen_display:	Decides whether to use screen (to display video being processed).
	"""
	video_name = path.split('/')[-1].split(".")[0]

	#stream = VideoStream(path)
	#stream.start()
	count = 0 # Counts number of mouth regions extracted

	for i in range(len(glob.glob('./frames/*.jpg'))): #find average width and height
		frame = cv2.imread('./frames/'+str(i+1)+'.jpg')
		rects = FACE_DETECTOR_MODEL(frame, 0)
		widths = []
		heights = []
		for rect in rects: #find avg width and height
			landmarks = LANDMARKS_PREDICTOR(frame, rect)
			mouth_coordinates = get_mouth_coord(landmarks)
			x, y, w, h = cv2.boundingRect(mouth_coordinates)
			widths.append(w)
			heights.append(h)
	avgwidth = round(sum(widths)/len(widths))
	avgheight = round(sum(heights)/len(heights))

	for i in range(len(glob.glob('./frames/*.jpg'))):
		#frame = stream.read()
		frame = cv2.imread('./frames/'+str(i+1)+'.jpg')

		#frame = resize(frame, IMAGE_WIDTH)

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
				name = output_dir +'/'+ str(count) + '.jpg',
				thewidth = avgwidth,
				theheight = avgheight)

			count+=1

		if screen_display:
			visualize(frame, all_mouth_coordinates)			

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	if screen_display:
		cv2.destroyAllWindows()


def getSEvideo(path, outputpath, numinterp): #input video relative path, output video relative path, number of smoothings by interpolation
	originalpath = path

	os.system('mkdir -p frames')
	os.system('mkdir -p audio')
	os.system('mkdir -p pictures')
	os.system('mkdir -p silentvid')
	os.system('mkdir -p pngfiles')

	video_name = path.split('/')[-1].split(".")[0]
	input_directory = '/'.join(path.split('/')[:-1])
	if path[-3:] != 'mp4': 
		os.system('ffmpeg -i '+path+' -q:v 0 '+input_directory+'/'+video_name+'.mp4')
		path = input_directory+'/'+video_name+'.mp4'

	#output_dir = outputdirectory
	outvideo_name = outputpath.split('/')[-1].split(".")[0]
	output_dir = '/'.join(outputpath.split('/')[:-1])

	load_trained_models()

	if not FACE_DETECTOR_MODEL:
		return False

	os.system('ffmpeg -i '+originalpath+' -qscale 0 frames/%d.jpg') #convert video to frames

	os.system('ffmpeg -i '+path+' -vn -acodec pcm_s16le -ar 44100 -ac 2 audio/'+video_name+'.wav') #extract audio from video, place in audio folder

	extract_mouth_regions(path, './pictures', screen_display=False) #pictures placed in pictures folder

	#smooth SE video using interpolationXn
	for k in glob.glob('pictures/*.jpg'): #copy files from pictures to png files and convert
		jpgim = Image.open(k)
		jpgim.save('pngfiles/'+k.split('/')[-1][:-4]+'.png')

	for ijk in range(int(numinterp)): #number of interpolations
		for i in list(range(len(glob.glob('pngfiles/*.png'))))[::-1]: #convert all number pictures to even numbers
			os.system('mv pngfiles/'+str(i)+'.png pngfiles/'+str(2*i)+'.png')
		for j in range(len(glob.glob('pngfiles/*.png'))-1): #smooth frames using interpolation
			os.system('python3 pytoflow/run.py --f1 pngfiles/'+str(2*j)+'.png --f2 pngfiles/'+str(2*(j+1))+'.png --o pngfiles/'+str(2*j+1)+'.png --task interp')

	#create SE video
	oldnumofpics = len(glob.glob('pictures/*.jpg'))
	newnumofpics = len(glob.glob('pngfiles/*.png'))
	framerate = (newnumofpics/oldnumofpics)*float(os.popen('ffmpeg -i '+path+' 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"').read()[:-1]) #as a float
	os.system('ffmpeg -framerate '+str(round(framerate))+' -start_number 0 -i pngfiles/%d.png -vcodec mpeg4 silentvid/'+video_name+'_silent.mp4') #silent SE video placed in 'silentvid folder
	#add audio to silent video
	os.system('ffmpeg -i silentvid/'+video_name+'_silent.mp4 -i audio/'+video_name+'.wav -c:v copy -c:a aac -strict experimental '+output_dir+'/'+outvideo_name+'.mp4') #only outputs to .mp4, you can change the extension afterwards if necessary

	os.system('rm -rf frames')
	os.system('rm -rf audio')
	os.system('rm -rf pictures')
	os.system('rm -rf silentvid')
	os.system('rm -rf pngfiles')

	return True

#if len(sys.argv[1:])<3:
#	print("**Error: no inputs. Correct usage for getSEvideo shown below:")
#	print("python3 getSEvideo.py 'input video relative-path-to-getSEvideo.py' 'output video relative-path-to-getSEvideo.py' num_of_temporal_smoothing_passes")
#	exit()

getSEvideo(sys.argv[1],sys.argv[2],sys.argv[3])
#dont bother automating the video recording, just do it on a phone
