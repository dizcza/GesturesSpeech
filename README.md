<html>
<head>
<script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script>
<h1 align="center">Sign language</h1>
</head>

<body>
<p align="center"><i>This repo provides the instruments for classification of signle signs and basic human emotions.</i><p>
<nav class="contents">
  <h2>Contents </h2>
  <ol>
    <li><a href="#info">Projects info</a></li>
	<li><a href="#art">State of art</a></li>
	<li><a href="#preprocess">Data pre-processing</a></li>
	<li><a href="#displacements">Body joint displacements</a></li>
	<li><a href="#weights">Body joint weights. Discriminant ratio</a></li>
	<li><a href="#scenario">The worst and the best testing scenarios</a></li>
	<li><a href="#wdtw">Weighted DTW comparison</a></li>
	<li><a href="#tools">Packages</a></li>
  </ol>
</nav>

<video controls>
  <source src="video_example.mp4" type="video/mp4">
</video>

<img src="png/anim.gif"/>

<h2 id="info">Projects info</h2>

<table style="width:100%; margin:0 auto">
	<tr>
		<th rowspan="2"></th>
		<th colspan="3">Data base</th>
	</tr>
	<tr>
		<td>MoCap</td>
		<td><a href="http://datascience.sehir.edu.tr/visapp2013">Kinect</a></td>
		<td>Emotion</td>
	</tr>
	<tr>
    	<td>contents</td>
		<td>common Ukrainian gestures</td>
		<td>basic hand motions</td>
		<td>facial emotions</td>
	</tr>
	<tr>
    	<td>data type</td>
		<td>c3d</td>
		<td>txt</td>
		<td>blend --> csv --> pkl</td>
	</tr>
	<tr>
    	<td>data dimension</td>
		<td>3D</td>
		<td>3D</td>
		<td>2D</td>
	</tr>
	<tr>
    	<td>data is measured in</td>
		<td>millimeters</td>
		<td>meters</td>
		<td>pixels</td>
	</tr>
	<tr>
    	<td># markers</td>
		<td>83</td>
		<td>20</td>
		<td>18</td>
	</tr>
	<tr>
    	<td># active markers</td>
		<td>50</td>
		<td>6</td>
		<td>18</td>
	</tr>
	<tr>
    	<td>FPS</td>
		<td>120</td>
		<td>30</td>
		<td>24</td>
	</tr>
	<tr>
    	<td>unique gestures</td>
		<td>139</td>
		<td>8</td>
		<td>9</td>
	</tr>
	<tr>
    	<td>samples per gesture</td>
		<td>2</td>
		<td>28</td>
		<td>3..27</td>
	</tr>
	<tr>
    	<td>train dataset size</td>
		<td>1 x 139 = 139</td>
		<td>8 x 8 = 64</td>
		<td>42</td>
	</tr>
	<tr>
    	<td>test dataset size</td>
		<td>1 x 139 = 139</td>
		<td>20 x 8 = 139</td>
		<td>36</td>
	</tr>
	<tr>
    	<td># actors</td>
		<td>1</td>
		<td>unknown</td>
		<td>3</td>
	</tr>
</table>
<ol>
<b>Notes</b>
	<li>The Kinect project is based on <a href="http://datascience.sehir.edu.tr/pub/VISAPP2013.pdf">this</a> paper and their <a href="http://datascience.sehir.edu.tr/visapp2013/WeightedDTW-Visapp2013-DB.rar">data</a> to compare its results with our MoCap project.</li>
	<li><i>Active markers</i> are the ones that carry information about the motion body parts. Hence we don't need to track them all. This parameter is set manually for each project.</li>
	<li>Number of <i>actors</i> is the number of humans, who were involved directly into data acquisition.</li>
	<li>For simplicity and generality, each facial emotion can be viewed and called as a gesture.</li>
</ol>


<h2 id="art">State of the art</h2>
<p>The main idea in gesture recognition is to maximize between-class variance <i>Db</i> and minimize within-class variance <i>Dw</i> by choosing appropriate hidden parameters (training step). For this purpose Weighted DTW algorithm has been <a href="http://datascience.sehir.edu.tr/pub/VISAPP2013.pdf">proposed</a>.</p>

<p>It's obvious, that a joint which is active in one gesture class may not be active in another gesture class. Hence weights have to be adjusted accordingly.
As Reyes et al. (2011) have observed, only six out of the 20 joints contribute in identifying a hand gesture: left hand, right hand, left wrist, right wrist, left elbow, right elbow. For example, for the right-hand-push-up gesture, one would expect the right hand, right elbow and right wrist joints to have large weights, but to have smaller weights for the left-hand-push-up gesture. We propose to use only 3 of them, w.r.t. to the left or right hand.</p>



<h2 id="preprocess">Data pre-processing</h2>
<p>Data pre-processing for MoCap and Kinect projects includes 2 steps:</p>
<ol>
    <li>Subtracting the shoulder center from all joints, which accounts for cases where the user is not in the center of the depth image.</li>
    <li>Normalizing the data with the distance between the left and the right shoulders to account for the variations due to the person's size.</li>
</ol>
<p>Additionally, besides these two steps, Emotion data pre-processing includes also slope aligning, Kalman filtering and dealing with eyes blinking.</p>



<h2 id="displacements">Body joint displacements</h2>
<p>Except datasets sizes, the main difference between MoCap and Kinect projects is the number of body joints (marker) that are active during the motion. Thus, Kinect project provides only 3 markers per hand while MoCap project operates with 50 (25 x 2) hand markers. Their contribution in the motion (or their activity) is shown below as a  joint's displacement sum over gesture frames (measured in normalized units) with right hand highlighted in light blue colour for Kinect sample and left hand - for MoCap sample, shown above as an animation of Ukrainian gesture "Добрий ранок". For Emotion project, a <i>smile</i> sample was taken with mouth markers highlighted in light blue.</p>
<table style="width:100%">
	<tr>
		<th>Kinect</th>
		<th>MoCap</th>
		<th>Emotion</th>
	<tr>
    <tr>
        <td>
            <img src="Kinect/png/joint_displacements.png"/>
        </td>
        <td>
            <img src="MOCAP/png/joint_displacements.png"/>
        </td>
		<td>
            <img src="Emotion/png/joint_displacements.png"/>
        </td>
    </tr>
</table>



<h2 id="weights">Body joint weights. Discriminant ratio</h2>
<p>Using the total displacement values of joints, the joint <i>j</i>'s weight value of class <i>g</i> is calculated via </p>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\omega_j^g = \frac{1-e^{-\beta D_j^g}}{\sum_{k}(1-e^{-\beta D_k^g})},"/>
</div>
<p>where <img src="http://latex.codecogs.com/gif.latex?D_j^g"/> is the <i>j'</i>s joint total displacement, averaged over all training samples in the gesture class "<i>g</i>", <img src="http://latex.codecogs.com/gif.latex?\beta"/> is a hidden parameter. </p>
<p>Total displacement of the joint "<i>j</i>" in one example is computed by</p>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?D_j=\sum_{i=2}^N \|  \overrightarrow{X}_i^j - \overrightarrow{X}_{i-1}^j  \|,"/></div>
<p>where <img src="http://latex.codecogs.com/gif.latex?\overrightarrow{X}_i^j = \big( x_i^j, y_i^j, z_i^j \big) "/> is a <i>j'</i>s joint position in the  frame "<i>i</i>".</p>

<p>Best <img src="http://latex.codecogs.com/gif.latex?\beta"/> yields the biggest discriminant ratio <img src="http://latex.codecogs.com/gif.latex?R = \rfrac{D_b}{/D_w}"/>. In our case, the <img src="http://latex.codecogs.com/gif.latex?R_{max}"/> obtains when  <img src="http://latex.codecogs.com/gif.latex?\beta"/> vanishes. That means</p>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\omega_j^g \xrightarrow{\beta \to 0} \frac{D_j^g}{\sum_{k}D_k^g}"/>
</div>

<table style="width:100%">
	<tr>
		<th>Kinect</th>
		<th>MoCap</th>
		<th>Emotion</th>
	<tr>
    <tr>
        <td>
            <img src="Kinect/png/choosing_beta.png"/>
        </td>
        <td>
            <img src="MOCAP/png/choosing_beta.png"/>
        </td>
		<td>
            <img src="Emotion/png/choosing_beta.png"/>
        </td>
    </tr>
</table>

<p>Note, that MoCap has only 1 training example and 1 testing example per unique gesture, while Kinect provides 20 training and 8 testing ones. Thus, we cannot compute the within-class variance for the MoCap project -- only between-class variance is availible for the discriminant ratio demonstration.</p>



<h2 id="scenario">The worst and the best testing scenarios</h2>
<p>The WORST test scenario is passed (with OK status) when the <b>max</b> DTW cost (or its modification) among the test sample "c" and all THE SAME class train samples <img src="http://latex.codecogs.com/gif.latex?q_i"/> is lower than the min DTW cost among the test sample "c" and all OTHER classes samples <img src="http://latex.codecogs.com/gif.latex?h_j"/>:</p>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\max_i WDTW(c,q_i) < \min_j WDTW(c,h_j)"/></div>
<p>The BEST test scenario is passed (with OK status) when the <b>min</b> DTW cost (or its modification) among the test sample "c" and all THE SAME class train samples <img src="http://latex.codecogs.com/gif.latex?q_i"/> is lower than the min DTW cost among the test sample "c" and all OTHER classes samples <img src="http://latex.codecogs.com/gif.latex?h_j"/>:</p>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\min_i WDTW(c,q_i) < \min_j WDTW(c,h_j)"/></div>
<div align="center">
<img src="png/scenarios.png" height="60%"/>
</div>
<p>It should be clear that the best scenario is also a <i>classic</i> scenario at finding the best fitted known pattern to unknown test sample.</p>



<h2 id="wdtw">Weighted DTW comparison</h2>
<p>When all hidden parameters are calculated and all weights are set for each gesture class, it's time to use WDTW to compare some unknown sequence (from a testing set) with a known one (from a training set).</p>
<p>Classical DTW algorithm takes <img src="http://latex.codecogs.com/gif.latex?O(n^2)"/> complexity both in time and space. Thus we speed up it to linear time and space complexity, using <a href="http://cs.fit.edu/~pkc/papers/tdm04.pdf">FastDTW</a> <a href="https://github.com/slaypni/fastdtw">implementation</a>. Although FastDTW carries disadvantage in worse accuracy, comparing to DTW, via changing controlling parameter, called radius <i>r</i>, it turns out, that in our case FastDTW yields the same performance as DTW does.</p>

<table style="width:100%">
	<tr>
		<th>Kinect</th>
		<th>MoCap</th>
	<tr>
    <tr>
        <td>
            <img src="Kinect/png/dtw_path.png"/>
        </td>
        <td>
            <img src="MOCAP/png/dtw_path.png"/>
        </td>
    </tr>
</table>

<p>Using Weighted FastDTW algorithm with only 6 crucial (both hands) body joints for Kinect project (with other weights set to zero), all testing gesture characters from the <a href="http://datascience.sehir.edu.tr/visapp2013/">database</a> were classified correctly, while simple (unweighted) FastDTW algorithm with the same 6 body joints yields 21.2% out-of-sample error in the best case scenario.</p>


<table style="width:100%">
	<caption><font size="2"><i>Single gesture recognition accuracy, % </i></font></caption>
	<tr>
		<th rowspan="3">Algorithm</th>
		<th colspan="6">Data base</th>
	</tr>
	<tr>
		<td colspan="2" align="center">MoCap</td>
		<td colspan="2" align="center">Kinect</td>
		<td colspan="2" align="center">Emotion</td>
	</tr>
	<tr>
    	<td>worst</td>
		<td>best</td>
		<td>worst</td>
		<td>best</td>
		<td>worst</td>
		<td>best</td>
	</tr>
	<tr>
    	<td>WDTW</td>
		<td>100</td>
		<td>100</td>
		<td>100</td>
		<td>100</td>
		<td>0</td>
		<td>80.6</td>
	</tr>
	<tr>
    	<td>DTW or FastDTW</td>
		<td>100</td>
		<td>100</td>
		<td>69.4</td>
		<td>78.8</td>
		<td>0</td>
		<td>83.3</td>
	</tr>
</table>


<p>At the same time, MoCap's simple DTW yields the same result (100% recognition accuracy) as the weighted one. It's because, firstly,  there is too much information per MoCap sample (too high FPS and too many markers) and, secondly, training and testing gestures were performed by the same skilled signer. Thus, training and testing examples are nearly identical.</p>
<p>WDTW algorithm correctly identified 29 / 36 emotions, while simple DTW identified 30 / 36. The difference in 1 correctly recognized sample doesn't make a weather. Nevertheless, the explanation lies in the variation of markers fasteting on the face, sensory noise and variation of facial expressions per unique emotion class.</p>



<h2 id="rrate">FPS dependency</h2>
<p>Another interesting observation shows that there is no need to use the whole dense data to be able to correctly classify it. For instance, using weighted DTW, setting FPS = 8 is enough for both projects data.</p>
<div align="center">
	<img src="png/error_vs_fps.png"/>
</div>



<h2 id="tools">Packages</h2>
<p>Free 3D Graphics Lab Motion Capture visualizers:</p>
<ul>
  <li><a href="http://b-tk.googlecode.com/svn/web/mokka/index.html">Mokka</a></li>
  <li><a href="http://www.blender.org/">Blender</a> (look at <a href="http://stackoverflow.com/questions/20499320/how-to-import-c3d-files-into-blender">here</a> to be able to import C3D files)</li>
  <li><a href="http://www.c-motion.com/free-downloads/">Free CMO Reader</a></li>
</ul>

<p>The current project is portable. Although it's built and maintained upon Python 3.4 x32 version, you can use 2.7.8 or higher 32 bit version of Python.</p>

<p>Obligatory Python </a> packages (can be found at <a href="http://www.lfd.uci.edu/~gohlke/pythonlibs">http://www.lfd.uci.edu/~gohlke/pythonlibs)</a>:</p>
<ul>
  <li><a href="https://github.com/EmbodiedCognition/py-c3d">c3d</a> to read and display c3d contents (<b>Note</b>: if you use Python 2.7.x, install also native <a href="http://code.google.com/p/b-tk/downloads/detail?name=python-btk-0.3.0_win32.exe">Biomechanical ToolKit</a>)</li>
  <li>for pretty animation like in OpenGL:
	<ul>
		<li><a href="http://github.com/lmjohns3/py-cli">climate</a></li>
		<li><a href="http://www.lfd.uci.edu/~gohlke/pythonlibs/#curses">curses</a></li>
		<li><a href="http://pyglet.readthedocs.org/en/pyglet-1.2-maintenance">pyglet</a></li>
	</ul>
  </li>
  <li><a href="http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.4.2/windows/matplotlib-1.4.2.win32-py2.7.exe/download">matplotlib</a> (with <i>pyparsing</i>, <i>dateutil</i>, <i>setuptools</i> and <i>six</i>)</li>
  <li><a href="http://sourceforge.net/projects/numpy/files/NumPy/1.9.1/numpy-1.9.1-win32-superpack-python2.7.exe/download">numpy</a></li>
  <li><a href="https://pypi.python.org/pypi/dtw/1.0">dtw</a> (dynamic time warping)</li>
  <li>(optional) <a href="http://sourceforge.net/projects/pywin32">win32com</a> module to work with Microsoft Excel files</li>
</ul>

</body>
</html>
