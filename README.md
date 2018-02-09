# acLSTM_motion
This folder contains an implementation of acRNN for the CMU motion database written in Pytorch.

See the following links for more background:

Auto-Conditioned Recurrent Networks for Extended Complex Human Motion Synthesis: https://arxiv.org/abs/1707.05363

CMU Motion Capture Database: http://mocap.cs.cmu.edu/

### Prequisite

You need to install python3.6 (python 2.7 should also be fine) and pytorch. You will also need to have transforms3d, which can be installed by using this command:
```
pip install transforms3d
```

### Data Preparation

To begin, you need to download the motion data form the CMU motion database in the form of bvh files. I have already put some sample bvh files including "salsa", "martial" and "indian" in the "train_data_bvh" folder.


Then to transform the bvh files into training data, go to the folder "code" and run `generate_training_data.py`. You will need to change the directory of the source motion folder and the target motioin folder on the last line. If you don't change anything, this code will create a directory "../train_data_xyz/indian" and generate the training data for indian dances in this folder.

### Training

After generating the training data, you can start to train the network by running the `pytorch_train_aclstm.py`. Again, you need to change some directories on the last few lines in the code, including "dances_folder" which is the location of the training data, "write_weight_folder" which is the location to save the weights of the network during training, "write_bvh_motion_folder" which is the location to save the temporate output of the network and the groundtruth motion sequences in the form of bvh, and "read_weight_path" which is the path of the network weights if you want to train the network from some pretrained weights other than from begining in which case it is set as "". If you don't change anything, this code will train the network upon the indian dance data and create two folders ("../train_weight_aclstm_indian/" and "../train_tmp_bvh_aclstm_indian/") to save the weights and temporate outputs.


### Testing

When the training is done, you can use `pytorch_test_synthesize_motion.py` to synthesize motions. You will need to change the last few lines to set the "read_weight_path" which is the location of the weights of the network you want to test, "write_bvh_motion_folder" which is the location of the output motions, "dances_folder" is the where the code randomly picked up a short initial sequence from. You may also want to set the "batch" to determine how many motion clips you want to generate, the "generate_frames_numbers" to determine the length of the motion clips et al.. If you don't change anything, the code will read the weights from the 86000th iteration and generate 5 indian dances in the form of bvh to "../test_bvh_aclstm_indian/". 

The output motions from the network usually have artifacts of sliding feet and sometimes underneath-ground feet. If you are not satisfied with these details, you can use `fix_feet.py` to solve it. The algorithm in this code is very simple and you are welcome to write a more complex version that can preserve the kinematics of the human body and share it to us.

For rendering the bvh motion, you can use softwares like MotionBuilder, Maya, 3D max or most easily, use an online BVH renderer for example:
http://lo-th.github.io/olympe/BVH_player.html 



Enjoy!
