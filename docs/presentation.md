---
author: "Vladislav Serkov"
title: "Traffic monitoring system"
footer: "somecompany challenge 2023 - Vladislav Serkov"
marp: true
theme: default
size: 16:9

style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  section {
    justify-content: start;    
  }
---

# Traffic monitoring system for counting incoming and outgoing vehicles

![bg right:50% h:125%](../outputs/clustered_traj.jpg)

---

<!-- paginate: true -->

## Implementation demonstration
<p align="center">
<video src="../outputs/cyberpunk.mp4" controls width="80%">
</p>

---

## System overview

<div class="mermaid">
graph LR
    subgraph Legend
        direction TB
        imp[Implemented]
        nimp{{Not implemented}}
    end
    input[Video feed]
    subgraph Offline
        direction LR
        roadplane[Road plane detection]
        lane{{Lane detection}}
        calib{{Sensor calibration}}
    end
    subgraph Online
        direction LR
        detect[Vehicle detection]
        track[Vehicle tracking]
        count[Vehicle counting]
        dir[Direction estimation]
        speed{{Speed estimation}}
    end
    output[(Tracks DB)]
    input --> Offline
    input --> detect
    roadplane --> detect
    lane --> count
    lane --> dir
    detect --> track
    track --> dir
    track --> count
    track --> speed
    calib --> speed
    dir --> output
    count --> output
    speed --> output
</div>

---

## Road plane segmentation

Well-researched topic with many both real-time and slower-than-real-time approaches.

<div class="columns">
<div>

###### Panoptic segmentation model
<video src="../outputs/cyberpunk.mp4" width="100%"></video>
I'm using a _SegFormer_ model from 🤗

</div>
<div>

###### Dataset

![bdd100k h:100%](https://github.com/bdd100k/bdd100k/blob/master/doc/images/teaser.gif?raw=true)
trained on _Cityscapes_ dataset.

</div>
<div>

---

## Road plane segmentation

### 💣🤯😭 Licences 😭🤯💣

**SegFormer** (https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
> 3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Blah blah ...

**Cityscapes** (https://www.cityscapes-dataset.com/license/)
> Blah blah ... you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain. Blah blah goes on...

**Bottom line**: Need to be careful with the choice of the model and dataset.

---

## Interlude: lane detection

<div class="columns">
<div>

- Could (and should) be the same segmentation model. *BDD100K* dataset includes lane annotations.
- Alternatively, lanes can be inferred from vehicle trajectories themselves. **Gotchas**:
    - Large memory footprint: stores all trajectories in memory.
    - Not real-time: uses clustering.
    - Unclear assignment in case of lane changes.

</div>
<div>

![Clustered trajectories](../outputs/clustered_traj.jpg)
_Almost all vehicle trajectories correctly clustered into 4 groups by lanes they use_

</div>
</div>

---

## Vehicle detection and tracking

- Basically a solved problem.
- Plethora of freely available models and datasets.
- Porting to different hardware is easy.
<br>

The implementation uses **YOLOv8n** model from Ultralytics trained on **COCO** dataset which includes cars, trucks, buses, etc.
MOT is performed using **ByteTrack**: good balance between speed and accuracy.

---

## Counting, direction and speed estimation

- **Counting** is supported by the tracking algorithm.
    - The algorithm assigns a unique ID to each detection.
    - Discarding spurious detections by enforcing minimum length of the trajectory (in number of frames).
    <br>
- **Direction** is estimated by comparing the first and the last position of the trajectory.
    - Assumes that the directions can be differentiated by either the X or Y axis.
    - More robust approach uses the info about the lane the vehicle is in.
    <br>
- For **speed estimation** depth estimation is the king.

---

## Counting, direction and speed estimation


<div class="columns">
<div>

###### Speed estimation

- Requires additional **knowledge** about the environment:
    - Camera in-/extrinsics
    - Scene depth
    - Reference object size for scale (optional)
- Or additional **hardware**:
    - Stereo camera
    - Radar/laser sensor

</div>
<div>

###### Monocular depth estimation

![Monocular depth estimation](https://github.com/TRI-ML/vidar/blob/main/media/figs/packnet.gif?raw=true)
*3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020, oral)*

</div>
</div>

---

## Counting, direction and speed estimation

Recent advances in monocular depth estimation: **Towards Zero-Shot Scale-Aware Monocular Depth Estimation (ICCV 2023)** by Toyota Research Institute (https://arxiv.org/abs/2306.17253)

> In this work we introduce ZeroDepth, a novel monocular depth estimation framework capable of predicting metric scale for arbitrary test images from different domains and camera parameters. ... achieved a new state-of-the-art in both settings using the same pre-trained model, outperforming methods that train on in-domain data and require test-time scaling to produce metric estimates


<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({ startOnLoad: true });

window.addEventListener('vscode.markdown.updateContent', function() { mermaid.init() });
</script>