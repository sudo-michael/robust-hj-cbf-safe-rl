# Safe Learning in the Real World via Adaptive Shielding with Hamilton-Jacobi Reachability
# Installation


# Computing BRTs
Computing the BRTs requires [optimized_dp](https://github.com/SFU-MARS/optimized_dp) to be installed, but is not required during training.

After installing `optimized_dp`, to compute the BRTs, simply run:
```
bash scripts/compute_brts.sh
```

# Turtlebot Traning
The method was tested with a [turtlebot2](https://www.turtlebot.com/turtlebot2/), using a local laptop for training  and using a Vicon motion capture system to track the location of the robot.
[ROS1 Noetic](https://wiki.ros.org/noetic) was used to communicate between all systems.
## On the Turtlebot
```
roscore
roslaunch turtlebot_bringup minimal.launch
```
## On the Training Laptop
```
roslaunch vicon_bridge vicon.launch
WANDB_MODE=offline python train_ros.py --config=examples/dropq_config.py 
```
# Acknowledgements
The RL implementations from [jaxrl](https://github.com/ikostrikov/jaxrl) / [jaxrl5](https://github.com/kylestach/fastrlap-release/tree/main/jaxrl5)