import btk
# The variable 'eventsFrame' contains regions of interest (ROI) set as sets of first and last frames
# Note: If you use NumPy from another algorithm to create ROIs
# be sure that the type is "int" or the same type than the architecture
# of your OS (int32 for 32-bit OS, int64 for 64-bit OS).
# Otherwise an error message will be triggered mentionning that the given type
# in eventsFrame[i][0] or eventsFrame[i][1] is not an "int".
eventsFrame = [[170, 190], [240, 260]]
# Read the file
reader = btk.btkAcquisitionFileReader()
reader.SetFilename('myFile.c3d')
reader.Update()
acq = reader.GetOutput()
writer = btk.btkAcquisitionFileWriter()
for i in range(0, len(eventsFrame)):
    # Copy original data
    clone = acq.Clone()
    # Crop the acquisition to keep only the ROI
    clone.ResizeFrameNumberFromEnd(acq.GetLastFrame() - eventsFrame[i][0] + 1)
    clone.ResizeFrameNumber(eventsFrame[i][1] - eventsFrame[i][0] + 1)
    clone.SetFirstFrame(eventsFrame[i][0])
    # Select only the events in the ROI
    clone.ClearEvents()
    for e in btk.Iterate(acq.GetEvents()):
        if ((e.GetFrame() > clone.GetFirstFrame()) and (e.GetFrame() < clone.GetLastFrame())):
            clone.AppendEvent(e)
    # Create new C3D file
    writer.SetInput(clone)
    writer.SetFilename('croppedRegion' + str(i + 1) + '.c3d')
    writer.Update()