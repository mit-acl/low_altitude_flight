# Low Altitude Path Planning
## Installation

### Steps
Creating a python virtual environment and install dependencies using requirements.txt.
Navigate to `imitation_learning` and install the low_altitude_nav package with
```sh
$ pip install -e .
```

## Running examples with the pretrained policy
Pretrained checkpoints with example data and the corresponding terrain is provided 
[here](https://drive.google.com/drive/folders/1lZB6FGuZ3LWLmVUV50NRfyZ7hF8ttvGC?usp=sharing).


To inspect the data and the pretrained policy, see examples in `imitation_learning/notebooks`.

To train a policy, use:
 ```sh
$ python3 scripts/train_resnet.py params/resnet.yaml
```

(Please change the paths involved accordingly.)

## Some Tips on Isaac Sim
To load a usd file into Isaac, use
```sh
omni.usd.get_context().open_stage("terrain.usd")
```
Camera pose can be set with
```sh
camera.set_world_poses(camera_positions, camera_orientations, convention='world')
```

### Related Works
This work builds on: \
[Safe Low-Altitude Navigation in Steep Terrain with Fixed-Wing Aerial Vehicles
](https://github.com/ethz-asl/terrain-navigation) \
[Learning High-Speed Flight in the Wild](https://github.com/uzh-rpg/agile_autonomy) \
[iPlanner: Imperative Path Planning](https://github.com/leggedrobotics/iPlanner) \
[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)