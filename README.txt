# Msc-tracking of small wildlife subject

Author:Jingyuan HE

This code has been made for my final year project with the aim of creating a real time tracker for small wildlife animals. 
For the purposes of this project some pre-record insects video are used for testing.

The main code is an improved KCF tracker. This can be found in tracker.py. 
The groundtruth value is get by GroundTruth.py.
The calculation of CLE and Overlap value is in calculateIOU.py

Code was also developed to evaluate this algorithm and compare it with others 
from the Main tracking API of OpenCV (MOSSE, KCF and CSRT). 

To run the tracker, all that needs to change is the path of the pre-recorded fileor use live feed from a camera., and run the run.py.

