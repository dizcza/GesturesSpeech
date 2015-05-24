# coding=utf-8

import bpy
import os
import numpy as np
import json

RAW_BLENDER_DIR = "raw_blend"


def prepare_folder(file_path):
    """
     Makes necessary folders.
    :param file_path: path .blend file
    :return: csv dir to dump data in
    """
    csv_global_dir = r"../csv"
    short_name = os.path.basename(file_path).split('.blend')[0]
    if not os.path.exists(csv_global_dir):
        os.makedirs(csv_global_dir)

    csv_short_dir = os.path.join(csv_global_dir, short_name)
    if not os.path.exists(csv_short_dir):
        os.makedirs(csv_short_dir)

    return csv_short_dir


def gather_labels():
    """
     Gathers and stores joint labels in order.
    """
    test_file_path = os.path.join(RAW_BLENDER_DIR, os.listdir(RAW_BLENDER_DIR)[0])
    try:
        bpy.ops.wm.open_mainfile(filepath=test_file_path)
    except RuntimeError:
        print("Invalid blend file path. Breaking down.")
        return
    D = bpy.data

    for clip in D.movieclips:
        for ob in clip.tracking.objects:
            labels = []
            for track in ob.tracks:
                labels.append(track.name)
            np.savetxt("valid_labels.txt", labels, fmt="%s")


def gather_fps():
    """
     Saves fps rate for each blend file.
    """
    fps = {}
    for fname in os.listdir(RAW_BLENDER_DIR):
        if fname.endswith(".blend"):
            fpath = os.path.join(RAW_BLENDER_DIR, fname)
            try:
                bpy.ops.wm.open_mainfile(filepath=fpath)
            except RuntimeError:
                print("Invalid blend file path. Breaking down.")
                return
            fps[fname] = bpy.context.scene.render.fps
    print(max(fps.values()), min(fps.values()))
    json.dump(fps, open("fps.json", 'w'))


def convert_tracking_data(file_path):
    """
     Converts a blend file into a csv format.
    :param file_path: path to .blend file
    """
    try:
        bpy.ops.wm.open_mainfile(filepath=file_path)
    except RuntimeError:
        print("Invalid blend file path. Breaking down.")
        return 
    D = bpy.data

    printFrameNums = False  # include frame numbers in the csv file
    relativeCoords = False  # marker coords will be relative to the dimensions of the clip

    csv_dir = prepare_folder(file_path)
    for clip in D.movieclips:
        width = clip.size[0]
        height = clip.size[1]
        for ob in clip.tracking.objects:
            for track in ob.tracks:
                fn = csv_dir + "/{0}.csv".format(track.name)
                with open(fn, 'w') as f:
                    framenum = 0
                    while framenum < clip.frame_duration:
                        markerAtFrame = track.markers.find_frame(framenum)
                        if markerAtFrame:
                            coords = markerAtFrame.co.xy
                        else:
                            coords = np.nan, np.nan
                        if relativeCoords:
                            if printFrameNums:
                                print('{0},{1},{2}'.format(framenum, coords[0], coords[1]), file=f)
                            else:
                                print('{0},{1}'.format(coords[0], coords[1]), file=f)
                        else:
                            if printFrameNums:
                                print('{0},{1},{2}'.format(framenum, coords[0] * width, coords[1] * height), file=f)
                            else:
                                print('{0},{1}'.format(coords[0] * width, coords[1] * height), file=f)
                        framenum += 1


def convert_all(directory):
    """
     Converts all blend files in the directory.
    :param directory: dir name with .blend files
    """
    for fname in os.listdir(directory):
        if fname.endswith(".blend"):
            path = os.path.join(directory, fname)
            convert_tracking_data(path)


def run():
    """
     Provides user interface for making a choice how to convert
      blend file(s) into a csv format.
    """
    print("\nRunning blend ---> csv converter...\n")
    msg = "Do you want to \n\t (1) convert specific .blend filename;\n"
    msg += "\t (2) convert all blender files in the '%s' folder? \nPress 'q' to quit." % RAW_BLENDER_DIR
    msg += "\n\nYOUR CHOICE: "
    user_choice = input(msg)
    if user_choice == "1":
        fname = input("Enter a filename: ")
        if not fname.endswith(".blend"):
            fname = fname.split('.')[0] + ".blend"
        path = os.path.join(RAW_BLENDER_DIR, fname)
        convert_tracking_data(path)
    elif user_choice == "2":
        convert_all(RAW_BLENDER_DIR)
    print("DONE.")


if __name__ == "__main__":
    warn = "(!) Blender files should lie in the /raw_blend/ directory."
    assert os.path.exists(RAW_BLENDER_DIR), warn
    run()
    os.system("pause")
