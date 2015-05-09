<html>
<head>
<h4 align="center">Blender to .csv converter</h4>
<style>
mark { 
    background-color: #E5E4E2;
    color: black;
}

</style>
</head>


<body>
<p>Instructions how to export motion tracking markers from .blend to .csv files.</p>
<ul>
	<li> "raw_blend" folder contains Blender files;</li>
	<li> "../csv" folder will be created after running the convert_to_csv.py script. It'll contain folders with names, taken from "raw_blend" folder, and each of them will contain csv files for each  tracking marker, presented in corresponding .blend file;</li>
	<li>each csv file corresponds to a particular marker (with the same name as created csv file has) and includes the information about Xs and Ys with comma delimeter, starting with a new line for each frame during the animation;</li>
	<li>"nan" values of Xs and/or Ys are created and stored, when the tracking marker disappears along the animation unless it shows up again.</li>
</ul>

<p>To run convert_to_csv.py sript you should:</p>
<ol>
	<li>Download and install <a href="https://www.blender.org">Blender</a>.</li>
	<li>Add <b>blender.exe</b>'s path to your Path variable -- should be like C:\Program Files\Blender Foundation\Blender.</li>
	<li>Put blender files into "raw_blend" folder -- this directory should lie with the convert_to_csv.py script.</li>
	<li>Run <mark>blender --background --python convert_to_csv.py</mark> command in cmd.</li>
</ol>

<p><b>Note.</b> After you successfully converted Blender files to .csv, to be able to run Emotion project you should either dump collected csv folders withs their files into pythonic pickle format -- pkl -- via <mark>dump_pickles()</mark> function in csv_reader.py module, OR donwload already pickled files from the "_data" folder and split them into Training and Testing folders via <mark>split_data()</mark> func, provided in csv_reader.py module. In both cases, you should manually set path to the  directory with pickled data. It can be done via changing the constant EMOTION_PATH_PICKLES in emotion.py script.</p>

</body>
</html>