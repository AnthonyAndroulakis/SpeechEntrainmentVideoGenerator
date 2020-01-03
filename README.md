# Speech Entrainment Video Generator
makes the task of generating Speech Entrainment videos easier

<div style="height: 0; padding-bottom: calc(100.00% + 35px); position:relative; width: 100%;"><iframe allow="autoplay; gyroscope;" allowfullscreen height="100%" referrerpolicy="strict-origin" src="https://www.kapwing.com/e/5e0ec6cd6025710014847670" style="border:0; height:100%; left:0; overflow:hidden; position:absolute; top:0; width:100%" title="Embedded content made with Kapwing" width="100%" /></div>

## how to use: 
1) record yourself saying something     
- for the best results, do the following:
a) have your entire face in the input video
b) reduce camera shaking
c) use a high-quality webcam
2) run the following script:    
```
python3 getSEvideo.py 'input video relative-path-to-getSEvideo.py' 'output video relative-path-to-getSEvideo.py' num_of_temporal_smoothing_passes
```

## example run:
`python3 getSEvideo.py input/input.MOV output/SEoutput.mp4 2` #note: the output video will always have a .mp4 extension. I did this to reduce possible errors. If you'd like to change video formats, I recommend using ffmpeg.

## requirements:
python2 or python3
- numpy
- dlib
- opencv-python
- vidstab

## optional prerequisite:
CUDA (will make this program run a lot faster)

## How it works:
1) video is inputted
2) video is stabilized using vidstab (https://github.com/AdamSpannbauer/python_video_stab)
3) face landmark points are extracted using dlib (http://dlib.net/python/index.html)
4) mouth is cropped from video frames (using util/data_preprocessing_autoencoder.py and util/video_stream.py from https://github.com/pandeydivesh15/AVSR-Deep-Speech)
5) extra (pictures) frames are added to smooth video (https://github.com/Coldog2333/pytoflow)

## Example input and output:
An example input video and a corresponding output video can be seen in the example_input_output folder. Because the person featured in these videos is myself, you may __not__ use these videos.

## References to code used:
- video stabilization: https://github.com/AdamSpannbauer/python_video_stab
- face landmark points: http://dlib.net/, https://github.com/pandeydivesh15/AVSR-Deep-Speech
- video temporal smoothing using interpolation: https://github.com/Coldog2333/pytoflow, https://github.com/anchen1011/toflow, http://toflow.csail.mit.edu/
```
@article{xue2019video,
  title={Video Enhancement with Task-Oriented Flow},
  author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  journal={International Journal of Computer Vision (IJCV)},
  volume={127},
  number={8},
  pages={1106--1125},
  year={2019},
  publisher={Springer}
}
```

## License:
You may use this code as long as you cite this repository and include in your references the codes I have under __References to code used__.
