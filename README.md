# SpeechEntrainmentVideoGenerator
makes the task of generating Speech Entrainment videos easier

## how to use: 
1) record yourself saying something (your entire face must be in the video in order to create the best results)
2) run the following script: python3 getSEvideo.py 'input video' 'output folder'

## example run:
python3 getSEvideo.py input/test.mp4 output #output SE video will be in the output directory and will be named testSE.mp4

### tip: use a better camera for better results. Using an iPhone will likely yield low-quality (grainy) videos.

## requirements:
- numpy
- dlib
- opencv-python
- vidstab

## todo:
apply https://github.com/anchen1011/toflow for final video smoothing
