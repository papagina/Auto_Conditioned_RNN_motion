
import numpy as np
import cv2 as cv
from cv2 import VideoCapture
import matplotlib.pyplot as plt
from collections import Counter

import transforms3d.euler as euler
import transforms3d.quaternions as quat

from pylab import *
from PIL import Image
import os
import getopt

import json # For formatted printing

import read_bvh_hierarchy

import rotation2xyz as helper
from rotation2xyz import *



def get_pos_joints_index(raw_frame_data, non_end_bones, skeleton):
    pos_dic=  helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    keys=OrderedDict()
    i=0
    for joint in pos_dic.keys():
        keys[joint]=i
        i=i+1
    return keys


def parse_frames(bvh_filename):
   bvh_file = open(bvh_filename, "r")
   lines = bvh_file.readlines()
   bvh_file.close()
   l = [lines.index(i) for i in lines if 'MOTION' in i]
   data_start=l[0]
   #data_start = lines.index('MOTION\n')
   first_frame  = data_start + 3
   
   num_params = len(lines[first_frame].split(' ')) 
   num_frames = len(lines) - first_frame
                                     
   data= np.zeros((num_frames,num_params))
   
   for i in range(num_frames):
       line = lines[first_frame + i].split(' ')
       line = line[0:len(line)]

       
       line_f = [float(e) for e in line]
       
       data[i,:] = line_f
           
   return data


standard_bvh_file="../train_data_bvh/standard.bvh"
weight_translation=0.01
skeleton, non_end_bones=read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)    
sample_data=parse_frames(standard_bvh_file)
joint_index= get_pos_joints_index(sample_data[0],non_end_bones, skeleton)

   
def get_frame_format_string(bvh_filename):
    bvh_file = open(bvh_filename, "r")
    lines = bvh_file.readlines()
    bvh_file.close()
    l = [lines.index(i) for i in lines if 'MOTION' in i]
    data_end=l[0]
    #data_end = lines.index('MOTION\n')
    data_end = data_end+2
    return lines[0:data_end+1]

def get_min_foot_and_hip_center(bvh_data):
    print (bvh_data.shape)
    lowest_points = []
    hip_index = joint_index['hip']
    left_foot_index = joint_index['lFoot']
    left_nub_index = joint_index['lFoot_Nub']
    right_foot_index = joint_index['rFoot']
    right_nub_index = joint_index['rFoot_Nub']
                
                
    for i in range(bvh_data.shape[0]):
        frame = bvh_data[i,:]
        #print 'hi1'
        foot_heights = [frame[left_foot_index*3+1],frame[left_nub_index*3+1],frame[right_foot_index*3+1],frame[right_nub_index*3+1]]
        lowest_point = min(foot_heights) + frame[hip_index*3 + 1]
        lowest_points.append(lowest_point)
        
                                
        #print lowest_point
    lowest_points = sort(lowest_points)
    num_frames = bvh_data.shape[0]
    quarter_length = int(num_frames/4)
    end = 3*quarter_length
    overall_lowest = mean(lowest_points[quarter_length:end])
    
    return overall_lowest

def sanity():
    for i in range(4):
        print ('hi')
        
 
def get_motion_center(bvh_data):
    center=np.zeros(3)
    for frame in bvh_data:
        center=center+frame[0:3]
    center=center/bvh_data.shape[0]
    return center
 
def augment_train_frame_data(train_frame_data, T, axisR) :
    
    hip_index=joint_index['hip']
    hip_pos=train_frame_data[hip_index*3 : hip_index*3+3]
    
    for i in range(int(len(train_frame_data)/3) ):
        if(i!=hip_index):
            train_frame_data[i*3: i*3+3]=train_frame_data[i*3: i*3+3]+hip_pos
    
    
    mat_r_augment=euler.axangle2mat(axisR[0:3], axisR[3])
    n=int(len(train_frame_data)/3)
    for i in range(n):
        raw_data=train_frame_data[i*3:i*3+3]
        new_data = np.dot(mat_r_augment, raw_data)+T
        train_frame_data[i*3:i*3+3]=new_data
    
    hip_pos=train_frame_data[hip_index*3 : hip_index*3+3]
    
    for i in range(int(len(train_frame_data)/3)):
        if(i!=hip_index):
            train_frame_data[i*3: i*3+3]=train_frame_data[i*3: i*3+3]-hip_pos
    
    return train_frame_data
    
def augment_train_data(train_data, T, axisR):
    result=list(map(lambda frame: augment_train_frame_data(frame, T, axisR), train_data))
    return np.array(result)
 

    
#input a vector of data, with the first three data as translation and the rest the euler rotation
#output a vector of data, with the first three data as translation not changed and the rest to quaternions.
#note: the input data are in z, x, y sequence
def get_one_frame_training_format_data(raw_frame_data, non_end_bones, skeleton):
    pos_dic=  helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    new_data= np.zeros(len(pos_dic.keys())*3)
    i=0
    hip_pos=pos_dic['hip']
    #print hip_pos

    for joint in pos_dic.keys():
        if(joint=='hip'):
            
            new_data[i*3:i*3+3]=pos_dic[joint].reshape(3)
        else:
            new_data[i*3:i*3+3]=pos_dic[joint].reshape(3)- hip_pos.reshape(3)
        i=i+1
    #print new_data
    new_data=new_data*0.01
    return new_data
    

 
def get_training_format_data(raw_data, non_end_bones, skeleton):
    new_data=[]
    for frame in raw_data:
        new_frame=get_one_frame_training_format_data(frame,  non_end_bones, skeleton)
        new_data=new_data+[new_frame]
    return np.array(new_data)




def get_weight_dict(skeleton):
    weight_dict=[]
    for joint in skeleton:
        parent_number=0.0
        j=joint
        while (skeleton[joint]['parent']!=None):
            parent_number=parent_number+1
            joint=skeleton[joint]['parent']
        weight= pow(math.e, -parent_number/5.0)
        weight_dict=weight_dict+[(j, weight)]
    return weight_dict



def get_train_data(bvh_filename):
    
    data=parse_frames(bvh_filename)
    train_data=get_training_format_data(data, non_end_bones,skeleton)
    center=get_motion_center(train_data) #get the avg position of the hip
    center[1]=0.0 #don't center the height

    new_train_data=augment_train_data(train_data, -center, [0,1,0, 0.0])
    return new_train_data
          

def write_frames(format_filename, out_filename, data):
    
    format_lines = get_frame_format_string(format_filename)

    
    num_frames = data.shape[0]
    format_lines[len(format_lines)-2]="Frames:\t"+str(num_frames)+"\n"
    
    bvh_file = open(out_filename, "w")
    bvh_file.writelines(format_lines)
    bvh_data_str=vectors2string(data)
    bvh_file.write(bvh_data_str)    
    bvh_file.close()

def regularize_angle(a):
	
	if abs(a) > 180:
		remainder = a%180
		print ('hi')
	else: 
		return a
	
	new_ang = -(sign(a)*180 - remainder)
	
	return new_ang

def write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, output_filename):
    bvh_vec_length = len(non_end_bones)*3 + 6
    
    out_data = np.zeros([len(xyz_motion), bvh_vec_length])
    for i in range(1, len(xyz_motion)):
        positions = xyz_motion[i]
        rotation_matrices, rotation_angles = helper.xyz_to_rotations_debug(skeleton, positions)
        new_motion1 = helper.rotation_dic_to_vec(rotation_angles, non_end_bones, positions)
								
        new_motion = np.array([round(a,6) for a in new_motion1])
        new_motion[0:3] = new_motion1[0:3]
								
        out_data[i,:] = np.transpose(new_motion[:,np.newaxis])
        
    
    write_frames(format_filename, output_filename, out_data)

def write_traindata_to_bvh(bvh_filename, train_data):
    seq_length=train_data.shape[0]
    xyz_motion = []
    format_filename = standard_bvh_file
    for i in range(seq_length):
        data = train_data[i]
        data = np.array([round(a,6) for a in train_data[i]])
        #print data
        #input(' ' )
        position = data_vec_to_position_dic(data, skeleton)        
        
        
        xyz_motion.append(position)

        
    write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, bvh_filename)
    
def data_vec_to_position_dic(data, skeleton):
    data = data*100
    hip_pos=data[joint_index['hip']*3:joint_index['hip']*3+3]
    positions={}
    for joint in joint_index:
        positions[joint]=data[joint_index[joint]*3:joint_index[joint]*3+3]
    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
            
    return positions
       
def get_pos_dic(frame, joint_index):
    positions={}
    for key in joint_index.keys():
        positions[key]=frame[joint_index[key]*3:joint_index[key]*3+3]
    return positions



#######################################################
#################### Write train_data to bvh###########                



def vector2string(data):
    s=' '.join(map(str, data))
    
    return s

def vectors2string(data):
    s='\n'.join(map(vector2string, data))
   
    return s
 
    
def get_child_list(skeleton,joint):
    child=[]
    for j in skeleton:
        parent=skeleton[j]['parent']
        if(parent==joint):
            child.append(j)
    return child
    
def get_norm(v):
    return np.sqrt( v[0]*v[0]+v[1]*v[1]+v[2]*v[2] )

def get_regularized_positions(positions):
    
    org_positions=positions
    new_positions=regularize_bones(org_positions, positions, skeleton, 'hip')
    return new_positions

def regularize_bones(original_positions, new_positions, skeleton, joint):
    children=get_child_list(skeleton, joint)
    for child in children:
        offsets=skeleton[child]['offsets']
        length=get_norm(offsets)
        direction=original_positions[child]-original_positions[joint]
        #print child
        new_vector=direction*length/get_norm(direction)
        #print child
        #print length, get_norm(direction)
        #print new_positions[child]
        new_positions[child]=new_positions[joint]+new_vector
        #print new_positions[child]
        new_positions=regularize_bones(original_positions,new_positions,skeleton,child)
    return new_positions

def get_regularized_train_data(one_frame_train_data):
    
    one_frame_train_data=one_frame_train_data*100.0
    positions={}
    for joint in joint_index:
        positions[joint]=one_frame_train_data[joint_index[joint]*3:joint_index[joint]*3+3]
    
    #print joint_index['hip']
    hip_pos=one_frame_train_data[joint_index['hip']*3:joint_index['hip']*3+3]

    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
            
    
    new_pos=get_regularized_positions(positions)
    
    
    new_data=np.zeros(one_frame_train_data.shape)
    i=0
    for joint in new_pos.keys():
        if (joint!='hip'):
            new_data[i*3:i*3+3]=new_pos[joint]-new_pos['hip']
        else:
            new_data[i*3:i*3+3]=new_pos[joint]
        i=i+1
    new_data=new_data*0.01
    return new_data

def check_length(one_frame_train_data):
    one_frame_train_data=one_frame_train_data*100.0
    positions={}
    for joint in joint_index:
        positions[joint]=one_frame_train_data[joint_index[joint]*3:joint_index[joint]*3+3]
    
    #print joint_index['hip']
    hip_pos=one_frame_train_data[joint_index['hip']*3:joint_index['hip']*3+3]

    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
    
    for joint in positions.keys():
        if(skeleton[joint]['parent']!=None):
            p1=positions[joint]
            p2=positions[skeleton[joint]['parent']]
            b=p2-p1
            #print get_norm(b), get_norm(skeleton[joint]['offsets'])
    
    

#print get_child_list(skeleton, 'rHand')
        
#bvh_data=get_bvh_format_data(new_train_data,w_dic,0.01)

#print "write_data"
#bvh_format=get_frame_format_string(bvh_file)

#write_bvh_file(martial_folder+"01_new.bvh",bvh_format,bvh_data)


#bvh_file="01.bvh"
#data=get_train_data(bvh_file)
#write_to_bvh_file("03_2.bvh", data)
#m=visualize_whole_seq(data[3689:3719])
#cv.imwrite("motion.png",m)

def make_data():
	folders = ["indian/"]

	for folder in folders:
    

	    bvh_folder="/home/zimo/Documents/ZY/Train_Data/train_data_bvh/"+ folder 
	    xyz_folder="/home/zimo/Documents/ZY/Train_Data/train_data_xyz_fixed/"+folder
	    dance_files=os.listdir(bvh_folder)
	    dances=[]
	    for dance_file in dance_files:
	        if os.path.isfile(xyz_folder+dance_file+".npy"):
	            continue;
	        print ("load "+dance_file)
	        dance=get_train_data(bvh_folder+dance_file)              
	        dances=dances+[dance]
	        print (dance)
	        input(' ')
	        #visualize_seq_to_video(xyz_folder+dance_file+".avi", dance)
	        #visualize_seq_to_bvh(xyz_folder+dance_file+".bvh", dance)
	        np.save(xyz_folder+dance_file, dance)

def save_mat_data(data, output_folder):
	for i in range(len(data['all_data'])):
		filename = output_folder + '/' + str(i) + '.npy'
		np.save(filename, data['all_data'][i][0])
		print (i)
		























