
from read_bvh import *
import numpy as np


bvh_file = '../test_bvh_aclstm_indian/out00.bvh'
train_data=read_bvh.get_train_data(bvh_file)
num_frames = train_data.shape[0]

joint_index = read_bvh.joint_index



hip_idx = 3 * joint_index['hip']
lfn_idx, rfn_idx = 3 * joint_index['lFoot_Nub'], 3 * joint_index['rFoot_Nub']

last_feet_pos = (train_data[1][lfn_idx:lfn_idx + 3], train_data[1][rfn_idx:rfn_idx + 3])
last_hip_pos = train_data[1][hip_idx:hip_idx + 3]
for i in xrange(2, num_frames):
    feet_pos = (train_data[i][lfn_idx:lfn_idx + 3], train_data[i][rfn_idx:rfn_idx + 3])
    hip_pos = train_data[i][hip_idx:hip_idx + 3]
    abs_lower_foot = feet_pos[lower_foot]+hip_pos
    if(abs_lower_foot[1]<0.01):
        set = (last_feet_pos[lower_foot] + last_hip_pos) - (feet_pos[lower_foot] + hip_pos)
        offset[1] = 0
        #train_data[i][hip_idx:hip_idx + 3] = train_data[i - 1][hip_idx:hip_idx + 3] + (hip_pos - last_hip_pos) + offset
        train_data[i][hip_idx:hip_idx + 3] += offset
        last_feet_pos = feet_pos
        #last_hip_pos = hip_pos
        last_hip_pos = train_data[i][hip_idx:hip_idx + 3]
	
for i in xrange(2, num_frames):
	feet_pos = (train_data[i][lfn_idx:lfn_idx + 3], train_data[i][rfn_idx:rfn_idx + 3])
	hip_pos = train_data[i][hip_idx:hip_idx + 3]
	lower_foot = (int)(feet_pos[0][1] > feet_pos[1][1])
	offset = np.zeros(3)
	abs_lower_foot = feet_pos[lower_foot]+hip_pos
	if (abs_lower_foot[1] < 0):
		offset[1]=-abs_lower_foot[1]  
	train_data[i][hip_idx:hip_idx + 3] += offset
    


read_bvh.write_traindata_to_bvh('../test_bvh_aclstm_indian/out00_feetfixed.bvh', train_data)
