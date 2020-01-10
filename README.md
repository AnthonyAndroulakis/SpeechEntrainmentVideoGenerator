# Speech Entrainment (SE) Video Generator
makes the task of generating Speech Entrainment videos easier       

![flowchart](https://github.com/AnthonyAndroulakis/SpeechEntrainmentVideoGenerator/blob/master/example_input_output/sevidgenchart.png)

## how to use: 
1) record yourself saying something     
- for the best results, do the following:
- 1) have your entire face in the input video
- 2) reduce camera shaking
- 3) use a high-quality webcam
2) run the following script:    
```
python3 getSEvideo.py 'input video path relative to getSEvideo.py' 'output video path relative to getSEvideo.py'
```

## example run:
`python3 getSEvideo.py input/input.MOV output/SEoutput.mp4` #note: the output video will always have a .mp4 extension. I did this to reduce possible errors. If you'd like to change video formats, I recommend using ffmpeg.

## requirements:
python2 or python3
- numpy
- dlib
- opencv-python
- pykalman
- pillow

#### note: the output video will not be playable in Quicktime since Quicktime will only play specific types of videos (https://stackoverflow.com/a/5220516). I avoided reencoding when possible in order to minimize quality-loss.

## How it works:
1) face landmark points are extracted using dlib (http://dlib.net/python/index.html)
2) mouth is cropped from video frames (using edited version of util/data_preprocessing_autoencoder.py from https://github.com/pandeydivesh15/AVSR-Deep-Speech)
3) mouth-bounding box is kalman filtered (https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data)
4) Speech Entrainment video is created using cropped mouth pictures and original audio

## Example input and output:
An example input video and a corresponding output video can be seen in the example_input_output folder. Because the person featured in these videos is myself, you may __not__ use these videos.

## References to code used:
- face landmark points: http://dlib.net/, https://github.com/pandeydivesh15/AVSR-Deep-Speech
- kalman filter: http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/, https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data

## License:
You may use this code as long as you cite this repository and include in your references the codes I have under __References to code used__.
