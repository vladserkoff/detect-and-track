# Traffic monitoring system for counting incoming and outgoing vehicles.

> The project will be using a fixed stationary camera with primary purpose of traffic flow monitoring in realtime. Some of the future expansion of this system could be: i) Vehicle Speed detection, ii) vehicle flow lane by lane(measure e-toll vs cash pay lane volume). This project has a big potential to be deployed in multiple sites if the first site installation goes successful.

## Assumptions

1. The system must be able to perform in real time. This means that the computational cost of the system must be low enough to be able to process the video stream in real time. 

1. Moreover, since the system is expected to be deployed en masse, the hardware should be relatively cheap and portable. This can be achieved by either running on a cheap low-power hardware, e.g. Jetson TX2, or by streaming the video to a remote server and running the system there. Both cases have their own pros and cons, and in this case we will assume that the system is running on an edge device.

1. Since the camera is stationary, the system can be divided into two parts: offline and online.
    * The offline part is performed once when the system is started. Since it doesn't need to be performed in real time, we have more freedom when selecting the approaches to use. The main concern here is to keep the memory footprint low enough to fit into the edge device RAM, while the CPU usage is not an issue.  This part consists of road plane and lane detection, and possibly camera calibration. The estimated parameters can be saved to a file and loaded when the system is started again.
    * The online part is performed in real time. It consists of vehicle detection, tracking, counting, and potentially speed estimation.


## High level system overview

### Road plane and lane detection

Road plane detection is used to discard spurious vehicle detections that are not on the main road. Lane detection is used to count vehicles lane by lane. Both tasks are using a neural network which are trained on datasets of road images, usually self-driving. The network outputs a binary mask of the road plane, which is then used to filter out the road plane from the image. Since the camera is stationary this needs to be done only once when the system is started.

### Vehicle detection

Unlike road plane detection, vehicle detection needs to be performed in real time. Depending on the place of the application and the hardware, this part of the system could be implemented either as a neural network or as a classical computer vision algorithm. The output of this part is a list of bounding boxes around the detected vehicles at each frame.

#### Neural network

If the hardware is powerful enough and/or the system is used in a critical location where there are people's lives at stake, a neural network approach should be used. For real time performance, the network architecture should be lightweight, e.g. MobileNet or YOLO. Pros: high accuracy under various conditions, possibility to detect different types of vehicles, possibility to detect objects other than vehicles, e.g. people or animals. Cons: requires powerful hardware, availability of training data.

#### Classical computer vision algorithm

If the hardware is not powerful enough and/or the system is used in a non-critical location, a classical computer vision algorithm can be used. One of the simplest ones is background subtraction. The algorithm is based on the assumption that the background is static and the foreground is moving. Pros: low computational cost, no need for training data. Cons: lower accuracy, doesn't differentiate between different types of objects.

### Vehicle tracking

Vehicle tracking is performed using real time object tracking algorithms. The input is a list of bounding boxes around the detected vehicles at each frame. The output is a list of vehicle tracks. The algorithm should be able to handle occlusions and track vehicles even when they are not detected for several frames.

### Vehicle counting

Vehicle counting is performed using the vehicle tracks. The algorithm should be able to count vehicles even when they are moving in groups or in opposite directions.

### Direction estimation

Direction estimation is performed using the vehicle tracks. The most simple approach, if the setup is known in advance, is to differentiate between vehicles going into opposite directions along one of the axes: left/right - X axis, in/out - Y axis. If the setup is not known in advance, the directions can be estimated by clustering the vehicle tracks and then comparing new tracks with previously detected clusters.

### Speed estimation

Speed estimation is a more challenging task that requires additional knowledge about the hardware, setup and environment. Possible approaches include: *i)* Using a deep learning based monocular depth estimation with depth scale calibration, *ii)* Using a stereo camera setup to triangulate the positions of the vehicles, *iii)* Using a radar/laser sensor. The approaches with multiple sensors require periodic or continuos self-calibration to account any drifts in the system.

#### Monocular depth estimation

Monocular depth estimation is a already a well researched topic with the main challenge being the calibration of the depth scale. This shortcoming has recently started to be addressed by the research community introducing scale-aware approaches, see [ZeroDepth](https://arxiv.org/abs/2306.17253).

#### Stereo camera setup

Stereo camera setup is a more robust approach that provides depth information directly. This requires accurate matching of the detections from different sensors, ideally concentrating smaller vehicle parts, e.g. license plates.

#### Radar/laser sensor

Radar/laser sensor is a more expensive approach that provides accurate depth information directly. 