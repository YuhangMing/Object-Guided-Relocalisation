import os
import sys
import numpy as np
import argparse
import math

def read_file_list(filename):
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='''
    This script computes the error between the ground truth pose and the estimated pose. 
    ''')
    parser.add_argument('folder', help='sequence folder')
    parser.add_argument('id', help='seuqnce id')
    args = parser.parse_args()
    seq_folder = str(args.folder)
    seq_id = str(args.id)

    # load GT from text file
    gt_dict = read_file_list("pose_info/GroundTruth/"+seq_folder+"0"+seq_id+".txt")
    f = open("pose_info/GroundTruth/"+ seq_folder +"0"+ seq_id +".txt", "a")
    for i in range(301):
        print(i)
        for frame_id, values in gt_dict.items():
            if frame_id < 1100:
                new_frame_id = frame_id - 986
            elif frame_id < 1400:
                new_frame_id = frame_id - 1186
            else:
                new_frame_id = frame_id - 1386
            
            if new_frame_id == i:
                print("%d, %d, %d", i, frame_id, new_frame_id)
                line = values[0] + " " + values[1] + " " + values[2] + " " + values[3] + " " + values[4] + " " + values[5] + " " + values[6]
                print(line)
                f.write(str(new_frame_id) + " " + line + "\n")
    f.close()

    # for name in file_name_list:
    #     f.write()
        
    #     f.write("  "+name+": frame_id, trans_err, rotation_err\n")
    #     if len(est_dict) == 0:
    #         f.write("        NO RESULT\n")
    #     else:
    #         avg_succ_t = 0.
    #         avg_succ_r = 0.
    #         count_succ = 0
    #         avg_fail_t = 0.
    #         avg_fail_r = 0.
    #         count_fail = 0
    #         count_frames = 0
    #         for frame_id, values in gt_dict.items():
    #             # if int(args.id) == 4 and frame_id in [1413, 1416]:
    #             #     continue
    #             # if int(args.id) == 8 and frame_id in [811]:
    #             #     continue
    #             count_frames = count_frames + 1
                
    #             if frame_id in est_dict:
    #                 est_val = est_dict[frame_id]
    #                 # estimate translation error
    #                 gt_xyz = np.array([float(val) for val in values[0:3]])
    #                 est_xyz = np.array([float(val) for val in est_val[0:3]])
    #                 diff = gt_xyz - est_xyz
    #                 te = np.linalg.norm(diff)
    #                 trans_err.append(te)
    #                 # estiamte rotation error
    #                 gt_quat = np.array([float(values[6]),float(values[3]),float(values[4]),float(values[5])])
    #                 est_quat = np.array([float(est_val[6]),float(est_val[3]),float(est_val[4]),float(est_val[5])])
    #                 re = rotation_difference(gt_quat, est_quat)
    #                 rot_err.append(re)
    #                 # write to file
    #                 f.write("    " + str(frame_id) + ",   " + str(te) + ", " + str(re) +"\n")
    #                 if te <= 0.05 and re <= 5:
    #                     avg_succ_t = avg_succ_t + te
    #                     avg_succ_r = avg_succ_r + re
    #                     count_succ = count_succ + 1
    #                 else:
    #                     avg_fail_t = avg_fail_t + te
    #                     avg_fail_r = avg_fail_r + re
    #                     count_fail = count_fail + 1
    #             else:
    #                 f.write("          " + str(frame_id) + ", xxxx, xxxx\n")
    #                 count_fail = count_fail + 1

    #             if count_frames%100==0:
    #                 f.write("AVG upon frame "+str(count_frames)+":\n")
    #                 if(count_succ>0):
    #                     f.write("    Succ="+str(count_succ)+", t=" +str(avg_succ_t/count_succ)+", r=" +str(avg_succ_r/count_succ)+"\n")
    #                 if(count_fail>0):
    #                     f.write("    Fail="+str(count_fail)+", t=" +str(avg_fail_t/count_fail)+", r=" +str(avg_fail_r/count_fail)+"\n")


    #         if count_succ > 0:
    #             avg_succ_t = avg_succ_t / count_succ
    #             avg_succ_r = avg_succ_r / count_succ
    #         if count_fail > 0:
    #             avg_fail_t = avg_fail_t / count_fail
    #             avg_fail_r = avg_fail_r / count_fail
    #         f.write("AVG overall:\n")
    #         f.write("    Succ="+str(count_succ)+", t=" +str(avg_succ_t)+", r=" +str(avg_succ_r)+"\n")
    #         f.write("    Fail="+str(count_fail)+", t=" +str(avg_fail_t)+", r=" +str(avg_fail_r)+"\n")

    #     print(trans_err)
    #     print(rot_err)
    #     print()
    #     trans_err_list.append(trans_err)
    #     rot_err_list.append(rot_err)
    # f.close()




    