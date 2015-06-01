<html>
<head>
<h4 align="center">Mocap project</h4>
</head>

<body>

<table style="width:100%">
  <tr>
    <th>files</th>
    <th>info</th>
  </tr>
  <tr>
    <td>mreader.py</td>
    <td>MoCap C3D files reader</td>
  </tr>
  <tr>
    <td>btk_fake.py</td>
    <td>provides all necessary functions to use C3d package as a native btk</td>
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
    <td>valid_labels.txt</td>
    <td>list of 83 valid body joints names</td>
  </tr>
  <tr>
    <td>MOCAP_INFO.json</td>
    <td>stores the weights and the last training results</td>
  </tr>
</table>


<p>Given C3D files have too large points frequency (FPS = 120). It's a matter of fact, that between class variance <i>Db</i> does not much depend of the chosen frequency: omitting 90% of data (take each 12'th frame) gives  nearly the same between class variance as raw data do. But we should not forget about the within class variance. Unfortunately, having only 1 training example per unique gesture, it's impossibly to compute this characteristic. It can turn out that with the FPS diminishing within class variance will go up, ipso facto reducing the discriminant ratio <i>R = Db / Dw</i>. </p>

</body>
</html>