# Learning Strategies with Double DQN and DDPG in the Atari and MuJoCo Environments
This repository contains the source code to reproduce the experiments in the reinforcement learning final project. 
<table>
  <tr>
    <td>
      <img src="figures/video.png" width="100">
    </td>
    <td>
      <img src="figures/break.png" width="100">
    </td>
    <td>
      <img src="figures/pong.png" width="100">
    </td>
    <td>
      <img src="figures/boxing.png" width="100">
    </td>
  </tr>
  <tr>
    <td>
      <img src="figures/hopper.png" width="100">
    </td>
    <td>
      <img src="figures/human.png" width="100">
    </td>
    <td>
      <img src="figures/cheetah.png" width="100">
    </td>
    <td>
      <img src="figures/ant.png" width="100">
    </td>
  </tr>
</table>

## Setup
Run the following command to build the environment:
```angular2html
conda env create -f environment.yml
conda activate rl
```

We conduct experiments in the Atari and MuJoCo environments, please refer to [Atari](https://www.gymlibrary.dev/environments/atari/) and [MuJoCo](https://www.gymlibrary.dev/environments/mujoco/) for more information about the environments.
## Experiments
Run the following command to train from scratch

```angular2html
python run.py --env_name $env_name --to_train
```
Run the following evaluate the ckeckpoint
```angular2html
python run.py --env_name $env_name 
```
Please guarantee your checkpoints are in the folder checkpoints/$env_name

We support the following environments:
- VideoPinball-ramNoFrameskip-v4
- BreakoutNoFrameskip-v4
- PongNoFrameskip-v4
- BoxingNoFrameskip-v4
- Hopper-v2
- Humanoid-v2
- HalfCheetah-v2
- Ant-v2

Our learning curve is as follows:
<table>
  <tr>
    <td>
      <img src="figures/VideoPinball-ramNoFrameskip-v4.png" width="400">
    </td>
    <td>
      <img src="figures/BreakoutNoFrameskip-v4.png" width="400">
    </td>
  </tr>
  <tr>
    <td>
      <img src="figures/PongNoFrameskip-v4.png" width="400">
    </td>
    <td>
      <img src="figures/BoxingNoFrameskip-v4.png" width="400">
    </td>
  </tr>
  <tr>
    <td>
      <img src="figures/Hopper-v2.png" width="400">
    </td>
    <td>
      <img src="figures/Humanoid-v2.png" width="400">
    </td>
  </tr>
  <tr>
    <td>
      <img src="figures/HalfCheetah-v2.png" width="400">
    </td>
    <td>
      <img src="figures/Ant-v2.png" width="400">
    </td>
  </tr>
</table>
                                                                                   