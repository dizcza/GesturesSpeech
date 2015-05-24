<html>
<head>
<h4 align="center">Blender to .csv converter</h4>
</head>


<body>
<p>Instructions how to export motion tracking markers from .blend to .csv files.</p>
<ul>
	<li> "raw_blend" folder contains Blender files;</li>
	<li> "../csv" folder will be created after running the <i>convert_to_csv.py</i> script. It'll contain folders with names, taken from "raw_blend" folder, and each of them will contain csv files for each  tracking marker, presented in corresponding .blend file;</li>
	<li>each csv file corresponds to a particular marker (with the same name as created csv file has) and includes the information about Xs and Ys with comma delimeter, starting with a new line for each frame during the animation;</li>
	<li>"nan" values of Xs and/or Ys are created and stored, when the tracking marker disappears along the animation unless it shows up again.</li>
</ul>

<p>To run <i>convert_to_csv.py</i> sript you should:</p>
<ol>
	<li>Download and install <a href="https://www.blender.org">Blender</a>.</li>
	<li>Add <b>blender.exe</b>'s path to your Path variable -- should be like C:\Program Files\Blender Foundation\Blender.</li>
	<li>Put blender files into <b>raw_blend</b> folder -- this directory should lie with the <i>convert_to_csv.py</i> script.</li>
	<li>Run <b>blender --background --python test.py</b> command in cmd to make sure you correctly set up Python and Blender environments.</li>
	<li>Run <b>blender --background --python convert_to_csv.py</b> command in cmd to begin converting blender files into csv.</li>
</ol>

<p><b>Note.</b> After you successfully converted Blender files to .csv, to be able to run Emotion project you should either dump collected csv folders withs their files into pythonic pickle format -- pkl -- via <mark>dump_pickles()</mark> function in csv_reader.py module, OR donwload already pickled files from the "_data" folder. In both cases, you should manually set a path to the  directory with pickled data. In order to do this, just change a constant EMOTION_PATH_PICKLES in emotion.py script.</p>

</body>
</html>