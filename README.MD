Source: https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Cpp

## Instructions (Linux / OSX)

Install [opencv for python](https://pypi.org/project/opencv-python/) in your environment
```
pip install opencv-python
pip install opencv-contrib-python
```

Load the car images you want to mask the license plates into the folder **LicPlateImages**.
Execute the script with the command
```
python Main.py
```

The results are collected within the folders **plate-removed** (images that were masked) and **to-check** (images that are to be checked because the program did not identify any license plate.
