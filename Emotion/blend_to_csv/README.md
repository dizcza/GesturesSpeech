### Blender to .csv converter

Instructions how to export motion tracking markers from .blend to .csv files:

*   "raw_blend" folder contains Blender files;
*   "../csv" folder will be created after running the ```convert_to_csv.py``` script. It'll contain folders with names, taken from "raw_blend" folder, and each of them will contain csv files for each tracking marker, presented in corresponding .blend file;
*   each csv file corresponds to a particular marker (with the same name as created csv file has) and includes the information about Xs and Ys with comma delimeter, starting with a new line for each frame during the animation;
*   "nan" values of Xs and Ys are created and stored, when the tracking marker disappears along the animation unless it shows up again.

To run ```convert_to_csv.py``` sript you should:

1.  Download and install [Blender](https://www.blender.org).
2.  Add **blender.exe**'s path to your Path variable -- should be like C:\Program Files\Blender Foundation\Blender.
3.  Put blender files into **raw_blend** folder -- this directory should lie with the ```convert_to_csv.py``` script.
4.  Run ```blender --background --python test.py``` command in cmd to make sure you correctly set up Python and Blender environments.
5.  Run ```blender --background --python convert_to_csv.py``` command in cmd to begin converting blender files into csv.

**Note.** After you successfully converted Blender files to .csv, to be able to run Emotion project you should either dump collected csv folders withs their files into pythonic pickle format -- pkl -- via ```dump_pickles()``` function in ```csv_reader.py``` module, OR donwload already pickled files from the "\_data" folder. In both cases, you should manually set a path to the directory with pickled data. In order to do this, just change a constant ```EMOTION_PATH_PICKLES``` in ```emotion.py``` script.