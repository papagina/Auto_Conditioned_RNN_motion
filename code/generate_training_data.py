import read_bvh
import numpy as np
from os import listdir
import os


def generate_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    print ("Generating training data for "+ src_bvh_folder)
    if (os.path.exists(tar_traindata_folder)==False):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names=listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len=len(bvh_dance_name)
        if(name_len>4):
            if(bvh_dance_name[name_len-4: name_len]==".bvh"):
                print ("Processing "+bvh_dance_name)
                dance=read_bvh.get_train_data(src_bvh_folder+bvh_dance_name)
                np.save(tar_traindata_folder+bvh_dance_name+".npy", dance)

def generate_bvh_from_traindata(src_train_folder, tar_bvh_folder):
    
    print ("Generating bvh data for "+ src_train_folder)
    if (os.path.exists(tar_bvh_folder)==False):
        os.makedirs(tar_bvh_folder)
    dances_names=listdir(src_train_folder)
    for dance_name in dances_names:
        name_len=len(dance_name)
        if(name_len>4):
            if(dance_name[name_len-4: name_len]==".npy"):
                print ("Processing"+dance_name)
                dance=np.load(src_train_folder+dance_name)
                dance2=[]
                for i in range(dance.shape[0]/8):
                    dance2=dance2+[dance[i*8]]
                print (len(dance2))
                read_bvh.write_traindata_to_bvh(tar_bvh_folder+dance_name+".bvh",np.array(dance2))
                


generate_traindata_from_bvh("../train_data_bvh/indian/","../train_data_xyz/indian/")
#generate_traindata_from_bvh("../train_data_bvh/salsa/","../train_data_xyz/salsa/")
#generate_traindata_from_bvh("../train_data_bvh/martial/","../train_data_xyz/martial/")