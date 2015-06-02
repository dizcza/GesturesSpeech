<html>
<head>
<h4 align="center">Emotion project</h4>
</head>

<body>

<table style="width:100%">
  <tr>
    <th>files</th>
    <th>info</th>
  </tr>

  <tr>
    <td>em_reader.py</td>
    <td>Emotion class for reading pickled data (refer to <a href="https://github.com/dizcza/GesturesSpeech/tree/dev/Emotion/_data">_data </a>directory)</td>
  </tr>
  <tr>
    <td>em_setting.py</td>
    <td>emotion: training (setting) best parameters</td>
  </tr>
  <tr>
    <td>em_testing.py</td>
    <td>emotion: testing part</td>
  </tr>
  <tr>
    <td>valid_labels.txt</td>
    <td>list of 18 valid face marker names</td>
  </tr>
  <tr>
    <td>demo_kalman.py</td>
    <td>demo of kalman filter tools</td>
  </tr>
  <tr>
    <td>EMOTION_INFO.json</td>
    <td>stores the weights and the last training results</td>
  </tr>

</table>


<img src="png/happy.png" height="400"/>

<p>Data preprocessing steps:</p>
<ol>
	<li> Subtracting nose position of the first frame from all markers.</li>
	<li> Diving data by the base line -- from jaw to eyebrows center.</li>
	<li> Slope aligning.</li>
	<li> Kalman filter data restoration (see left picture below).</li>
	<li> Dealing with eyes winking (see right picture below).</li>
</ol>

<table>
	<tr>
		<td><img src="png/kalman.png"/></td>
		<td><img src="png/winking.png"/></td>
	</tr>
</table>

<p style="clear: both;">The data were divided into training (42 samples) and testing (36 samples) sets with unproprotional number of samples for each emotion.
</p>

</body>
</html>