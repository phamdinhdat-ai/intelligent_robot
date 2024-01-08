import numpy as np 
import os 


def cal_orientation(H):
    """
    H is the Hartenbeg matrix before multiplying transformation matrix
    """
    
    o_z = np.arctan2(H[0, 0], H[1, 0])
    o_y = np.arctan2(-H[2,0], H[0,0]*np.cos(o_z) + H[1, 0] * np.sin(o_z))
    o_x = np.arctan2(H[0, 2]*np.sin(o_z) - H[1,2]*np.cos(o_z), H[1,2]*np.cos(o_z) - H[0,2]* np.sin(o_z))
    return o_x, o_y, o_z