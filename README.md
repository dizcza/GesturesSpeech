<html>
<head>
<h3 align="center">GesturesSpeech</h3>
</head>

<body>
<p align="center"><i>This repo provides the intruments to work with Motion Capture .c3d files, which includes Ukrainian gestures.</i></p>

<video src="video_example.mp4" width="320" height="200" controls preload></video>

<table style="width:100%">
  <tr>
    <th>script</th>
    <th>file info</th>	
  </tr>
  <tr>
    <td>files_modifier.py</td>
    <td>changes orientation XYZ by default (Z should measure human height)</td>	
  </tr>
  <tr>
    <td>helper.py</td>
    <td>prints info about open .c3d file</td>
  </tr>
  <tr>
    <td>labeling.py</td>
    <td>contains all 83 markers (labels) names with init (relaxed) frame for each .c3d file</td>
  </tr>
  <tr>
    <td><b><i>main.py<b></i></td>
    <td><i>main file to work with .c3d files</i></td>
  </tr>
  <tr>
    <td>math_kernel.py</td>
    <td>provides necessary math function</td>
  </tr>
  <tr>
    <td>plotter.py</td>
    <td>displays animation</td>
  </tr>
</table> 

<p>Free visualizers:</p>
<ul>
  <li><a href="http://b-tk.googlecode.com/svn/web/mokka/index.html">Mokka</a></li>
  <li><a href="http://www.c-motion.com/free-downloads/">Free CMO Reader</a></li>
</ul>

<p>Obligatory <a href="https://www.python.org/ftp/python/2.7/python-2.7.msi">Python 2.7</a> packages (can be found at http://www.lfd.uci.edu/~gohlke/pythonlibs/):</p>
<ul>
  <li> <a href="http://code.google.com/p/b-tk/downloads/detail?name=python-btk-0.3.0_win32.exe">The Biomechanical ToolKit</a>
  		to read and modify data from .c3d files</li>
  <li> <a href="http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.4.2/windows/matplotlib-1.4.2.win32-py2.7.exe/download">matplotlib</a> (with <b>pyparsing</b>, <b>dateutil</b>, <b>pytz</b> and <b>six</b>)</li>
  <li><a href="http://sourceforge.net/projects/numpy/files/NumPy/1.9.1/numpy-1.9.1-win32-superpack-python2.7.exe/download"> 		numpy</a></li>
</ul>

</body>
</html>
