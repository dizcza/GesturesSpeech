# Sign language recognition

###### This repo provides the instruments for single signs and facial emotions recognition.


## Quick start

Inside `GesturesSpeech` (repo's root) type:

1. `pip install -r requirements.txt`
2. check one project at a time:
	* `python -m MOCAP.mreader`
	* `python -m Kinect.kreader`
	* `python -m Emotion.em_reader`

For an in-depth start see [how to use a project](how-to-use.md).

For gained recognition accuracy go to [results](results) section. 

#### Tested on:
* Windows 7 SP1 x64 + Python 3.4.3 x32 (also with Python 2.7.10 x32)
* Ubuntu 14.04.3 x64 + Python 2.7.6 x64 (built-in)


## Dependencies

*   [numpy](http://sourceforge.net/projects/numpy)
*   [c3d](https://github.com/EmbodiedCognition/py-c3d) to read and display c3d contents (Note: if you use Python 2.7, install also native [Biomechanical ToolKit](https://code.google.com/p/b-tk/downloads/list))
*   [pyglet](http://pyglet.readthedocs.org) for OpenGL graphics
*   [matplotlib](http://sourceforge.net/projects/matplotlib) for scientific results
*   [dtw](https://pypi.python.org/pypi/dtw) [dynamic time warping] is only for illustration purpose (I used [FastDTW](https://github.com/slaypni/fastdtw) modification)
*   [tqdm](https://github.com/tqdm/tqdm) progress bar
*   [rarfile](https://github.com/markokr/rarfile) to unrar automatically downloaded Kinect database