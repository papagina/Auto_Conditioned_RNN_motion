
import read_bvh
import numpy as np


bvh_file = 'xx.bvh'
train_data=read_bvh.get_train_data(bvh_file)
train_data = train_data[1:]
num_frames = train_data.shape[0]

joint_index = read_bvh.joint_index



hip_idx = 3 * joint_index['hip']
lfn_idx, rfn_idx = 3 * joint_index['lFoot_Nub'], 3 * joint_index['rFoot_Nub']

print ("Find the ground level.")
global_feet_y =np.concatenate(( train_data[:, lfn_idx+1]+train_data[:, hip_idx+1 ], train_data[:, rfn_idx+1]+train_data[:, hip_idx+1 ]), 0)
global_feet_y=np.sort(global_feet_y)
ground_level = global_feet_y[0:int(num_frames/4)].mean()

print ("The ground level is: "+str(ground_level))

print ("Move the ground to 0.")
train_data[:, hip_idx+1 ] +=-ground_level


print ("Solve feet sliding.")

last_feet_pos = (train_data[0, lfn_idx:lfn_idx + 3], train_data[0, rfn_idx:rfn_idx + 3])

hip_pos_v = train_data[1:, hip_idx:hip_idx+3] - train_data[0:num_frames-1, hip_idx:hip_idx+3] ##difference of hips between frames. 
is_last_frame_feet_on_ground = [0,0] # left, right
for i in range(1, num_frames-1):
    hip_pos = train_data[i, hip_idx:hip_idx + 3]
    feet_pos = (train_data[i,lfn_idx:lfn_idx + 3], train_data[i,rfn_idx:rfn_idx + 3])
    lower_foot_idx = (int)(feet_pos[0][1] > feet_pos[1][1])
    global_lower_foot = feet_pos[lower_foot_idx]+hip_pos

    if(global_lower_foot[1]<0.01): ##If touches the ground
        if(is_last_frame_feet_on_ground[lower_foot_idx] == 1):
            offset = (last_feet_pos[lower_foot_idx] - feet_pos[lower_foot_idx]) 
            hip_pos_v[i-1,0] =offset[0]
            hip_pos_v[i-1,2] =offset[2]
        is_last_frame_feet_on_ground=[0,0]
        is_last_frame_feet_on_ground[lower_foot_idx]=1
        print ("fix frame "+str(i))
    else:
        is_last_frame_feet_on_ground=[0,0]
        
    last_feet_pos = feet_pos

for i in range(1, num_frames):   
    train_data[i,hip_idx:hip_idx+3] = train_data[i-1,hip_idx:hip_idx+3] + hip_pos_v[i-1] 
    

    
print ("write bvh")

read_bvh.write_traindata_to_bvh('xx_fixed.bvh', train_data)





