<html>
<head>
<h3 align="center">GesturesSpeech</h3>
</head>

<body>
<p align="center"><i>This repo provides the instruments to work with Motion Capture data, which include Ukrainian gestures.</i></p>

<video src="video_example.mp4" width="320" height="200" controls preload></video>

<p>Projects kernel summary:</p>
<table style="width:100%">
  <tr>
    <th>script</th>
    <th>file info</th>
  </tr>
  <tr>
    <td>humanoid</td>
    <td>HumanoidBasic class: super class for both projects</td>
  </tr>
  <tr>
    <td>gDTW</td>
    <td>gesture weighted DTW algorithm implementation</td>
  </tr>
</table>


<h6>Projects info:</h6>
<ul>
  <li>MoCap (Motion Capture):
    <ul>
      <li>covers common Ukrainian gestures</li>
      <li>file type: c3d</li>
      <li>body joints: 83</li>
      <li>data is measured in millimeters</li>
      <li>FPS: 120</li>
      <li>unique gesture classes: ~150</li>
      <li>samples per gesture: 2</li>
    </ul>
  </li>
  <li><a href="http://datascience.sehir.edu.tr/visapp2013">Kinect<a/>:</li>
    <ul>
      <li>covers basic hand motions</li>
      <li>file type: txt</li>
      <li>body joints: 20</li>
      <li>data is measured in meters</li>
      <li>FPS: 30</li>
      <li>unique gesture classes: 8</li>
      <li>samples per gesture: 28</li>
    </ul>
</ul>

<table style="width:100%">
  <tr>
    <th>Project</th>
    <th>script</th>
    <th>file info</th>	
  </tr>
  
  <tr>
    <td rowspan="10">MoCap</td>
    <td>files_modifier.py</td>
    <td>changes orientation XYZ by default (Z should measure human height)</td>	
  </tr>
  <tr>
    <td>mreader.py</td>
    <td>MoCap C3D files reader</td>
  </tr>
  <tr>
    <td>msetting.py</td>
    <td>MoCap training part</td>
  </tr>
  <tr>
    <td>mtesting.py</td>
    <td>MoCap testing part</td>
  </tr>
  <tr>
    <td>helper.py</td>
    <td>prints info about open .c3d file</td>
  </tr>
  <tr>
    <td>labelling.py</td>
    <td>contains all 83 markers (labels) names with init (relaxed) frame for each .c3d file</td>
  </tr>
  <tr>
    <td>math_kernel.py</td>
    <td>provides necessary math function</td>
  </tr>
  <tr>
    <td>splitter.py</td>
    <td>HumanoidUkrSplitter class implementation</td>
  </tr>
  <tr>
    <td>troubles_hunter.py</td>
    <td>checks split C3D files for having troubles and shows them if any</td>
  </tr>
  <tr>
    <td>valid_labels.txt</td>
    <td>list of 83 valid body joints names</td>
  </tr>  
  
  <tr>
    <td rowspan="4">Kinect</td>
    <td>kreader.py</td>
    <td>Kinect txt files reader</td>
  </tr>
  <tr>
    <td>ksetting.py</td>
    <td>Kinect training part</td>
  </tr>
  <tr>
    <td>ktesting.py</td>
    <td>Kinect testing part</td>
  </tr>
  <tr>
    <td>KINECT_INFO.json</td>
    <td>includes the necessary info and chosen params for the Kinect project</td>
  </tr>
  
</table> 

<p>Free 3D Motion Capture visualizers:</p>
<ul>
  <li><a href="http://b-tk.googlecode.com/svn/web/mokka/index.html">Mokka</a></li>
  <li>A powerfull <a href="http://www.blender.org/">Blender</a> suite (look at <a href="http://stackoverflow.com/questions/20499320/how-to-import-c3d-files-into-blender">here</a> to be able to import .c3d files)</li>
  <li><a href="http://www.c-motion.com/free-downloads/">Free CMO Reader</a></li>
</ul>

<p>Obligatory <a href="https://www.python.org/ftp/python/2.7/python-2.7.msi">Python 2.7</a> packages (can be found at http://www.lfd.uci.edu/~gohlke/pythonlibs/):</p>
<ul>
  <li><a href="http://code.google.com/p/b-tk/downloads/detail?name=python-btk-0.3.0_win32.exe">The Biomechanical ToolKit</a>
  		to read and modify data from .c3d files</li>
  <li><a href="http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.4.2/windows/matplotlib-1.4.2.win32-py2.7.exe/download">matplotlib</a> (with <b>pyparsing</b>, <b>dateutil</b>, <b>pytz</b> and <b>six</b>)</li>
  <li><a href="http://sourceforge.net/projects/numpy/files/NumPy/1.9.1/numpy-1.9.1-win32-superpack-python2.7.exe/download"> 		numpy</a></li>
  <li><a href="https://pypi.python.org/pypi/dtw/1.0">dtw</a> (dynamic time warping)</li>
</ul>

</body>
</html>
