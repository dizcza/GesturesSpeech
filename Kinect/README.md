<html>
<head>
<h4 align="center">Kinect project</h4>
</head>

<body>
<p>The main idea: a joint that is active in one gesture class may not be active in another gesture class. Hence weights has to be adjusted accordingly.</p>
<p>As <a href="http://datascience.sehir.edu.tr/pub/VISAPP2013.pdf">they</a> have observed, only six out of the 20 joints contribute in identifying a hand gesture: left hand, right hand, left wrist, right wrist, left elbow, right elbow. We propose to use only 3 of them, w.r.t. to the left or right hand.</p>

<p>For example, for the right-hand-push-up gesture, one would expect the right hand, right elbow and right wrist joints to have large weights, but to have smaller weights for the left-hand-push-up gesture.</p>
<div><img src="joint_displacements.png" height="300"/></div>
<p>When all weights are set for each gesture class, it's time to use DTW to compare some unknown sequence (from a testing set) with a known one (from a training set).</p>
<div><img src="dtw_path.png" height="300"/></div>
<p>Using only 3 crucial body joints (with the other weights to be zeros), all testing gesture characters from the <a href="http://datascience.sehir.edu.tr/visapp2013/">database</a> are classified correctly.</p>

</body>
</html>
