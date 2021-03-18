import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse
import math

def load_pose_from_file(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(int(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    This script computes the error between the ground truth pose and the estimated pose. 
    ''')
    parser.add_argument('folder', help='sequence folder')
    parser.add_argument('id', help='seuqnce id')
    args = parser.parse_args()
    seq_folder = str(args.folder)
    seq_id = str(args.id)

    construction_seq = load_pose_from_file("build/map-" + seq_folder + "0" + seq_id + ".data-kfs.txt")
    reloc_seq = load_pose_from_file("pose_info/CENT/" + seq_folder + "0"+ seq_id + ".txt")

    # get trans and draw
    num_const_pose = len(construction_seq)
    const_x = np.zeros(num_const_pose)
    const_y = np.zeros(num_const_pose)
    const_z = np.zeros(num_const_pose)
    count = 0
    for frame_id, pose in construction_seq.items():        
        const_x[count] = pose[0]
        const_y[count] = pose[1]
        const_z[count] = pose[2]
        count = count + 1

    num_reloc_pose = len(reloc_seq)
    reloc_x = np.zeros(num_reloc_pose)
    reloc_y = np.zeros(num_reloc_pose)
    reloc_z = np.zeros(num_reloc_pose)
    count = 0
    for frame_id, pose in reloc_seq.items():
        if count == 0:
            reloc_x[count] = pose[0]
            reloc_y[count] = pose[1]
            reloc_z[count] = pose[2]
        else:
            # print(type(reloc_x[count-1]))
            # print(type(pose[0]))
            diff_x = reloc_x[count-1]-float(pose[0])
            diff_y = reloc_y[count-1]-float(pose[1])
            diff_z = reloc_z[count-1]-float(pose[2])
            dist = math.sqrt( diff_x**2 + diff_y**2 + diff_z**2 )
            # print(dist)
            if dist < 0.1:
                reloc_x[count] = pose[0]
                reloc_y[count] = pose[1]
                reloc_z[count] = pose[2]
            else:
                reloc_x[count] = reloc_x[count-1]
                reloc_y[count] = reloc_y[count-1]
                reloc_z[count] = reloc_z[count-1]
        count = count + 1
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(const_x, const_y, const_z, 'gray')
    # hold on
    ax.plot3D(reloc_x, reloc_y, reloc_z, 'red')
    plt.show()
    

