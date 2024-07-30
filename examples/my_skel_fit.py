# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import pickle

import trimesh
import torch
import sys
import mmap
sys.path.append("../")

from skel.alignment.aligner import SkelFitter
from skel.alignment.utils import load_smpl_seq

max_size = 1024 * 1024 * 10

import json, mmap, numpy as np
def read_parameters():
    global max_size
    file_path = '/home/mmap/mmap_smpl_params.txt'
    while True:
        
        # check that the file contains data
        if not os.path.exists(file_path):
            continue
        if os.path.getsize(file_path) == 0:
            continue
        
        size = min(max_size, os.path.getsize(file_path))
        with open(file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), size)
            mm.seek(0)
            json_length_bytes = mm.read(4)
            if not json_length_bytes:
                mm.close()
                continue
            
            json_length = int.from_bytes(json_length_bytes, byteorder='little')
            json_bytes = mm.read(json_length)
            mm.close()
        
        json_bytes = json_bytes.decode("utf-8").rstrip('\0')
        
        try:
            data = json.loads(json_bytes)
            print("Received: ", data)
            f.close()
            return data
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")


# def send_skel(skel_mesh):
#     global max_size
#     file_path = '/home/mmap/mmap_skel.txt'
#     with open(file_path, 'w+b') as f:
#         # TODO FIX LENGTH
#         size = min(max_size, len(skel_mesh))
#         print("Sending size: ", size)
#         mm = mmap.mmap(f.fileno(), size)
#         mm.seek(0)
#         # write size
#         mm.write(len(skel_mesh).to_bytes(4, byteorder='little'))
#         mm.write(skel_mesh)
#         mm.close()

def send_skel(obj_data):
    global max_size
    file_path = '/home/mmap/mmap_skel.txt'
    flag_path = '/home/mmap/mmap_skel_flag.txt'
    
    # Ensure the mmap file is the right size
    with open(file_path, 'wb') as f:
        f.write(b'\x00' * len(obj_data))
    
    with open(file_path, 'r+b') as f:
        # Memory-map the file, size 0 means whole file
        mm = mmap.mmap(f.fileno(), 0)
        
        # Write the .obj file data to the mmap file
        mm.write(obj_data.encode('utf-8'))
        
        # Close the memory-mapped file
        mm.close()
                # Signal that new data is ready
    # simple way to synchronize docker and host
    with open(flag_path, 'w') as flag_file:
        flag_file.write('1')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL frame')
    
    # parser.add_argument('--smpl_mesh_path', type=str, help='Path to the SMPL mesh to align to', default=None)
    # parser.add_argument('--smpl_data_path', type=str, help='Path to the SMPL dictionary to align to (.pkl or .npz)', default=None)
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='output')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', action='store_true')
    # parser.add_argument('--gender', type=str, help='Gender of the subject (only needed if not provided with smpl_data_path)', default='female')
    
    args = parser.parse_args()
    args.smpl_data_path = 'examples/samples/img_fit/emily-sea-coiWR0gT8Cw-unsplash_0.npz'
    # smpl_data = load_smpl_seq(args.smpl_data_path, gender=args.gender, straighten_hands=False)
    
    # if args.smpl_mesh_path is not None:
    #     subj_name = os.path.basename(args.smpl_seq_path).split(".")[0]
    if args.smpl_data_path is not None:
        subj_name = os.path.basename(args.smpl_data_path).split(".")[0]
    else:
        raise ValueError('Either smpl_mesh_path or smpl_data_path must be provided')
    
    # Create the output directory
    subj_dir = os.path.join(args.out_dir, subj_name)
    os.makedirs(subj_dir, exist_ok=True)
    pkl_path = os.path.join(subj_dir, subj_name+'_skel.pkl')  
    
    subj_dir = subj_dir
    
    if False: #os.path.exists(pkl_path) and not args.force_recompute:
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(subj_dir))
        skel_data_init = pickle.load(open(pkl_path, 'rb'))
    else:
        skel_data_init = None
    while True:
        data = read_parameters()

        # skel_fitter = SkelFitter(smpl_data['gender'], device='cuda:0', export_meshes=True)
        skel_fitter = SkelFitter(data["gender"], device='cuda:0', export_meshes=True)
        
        # skel_seq = skel_fitter.run_fit(smpl_data['trans'], 
        #                            smpl_data['betas'], 
        #                            smpl_data['poses'],
        #                            batch_size=1,
        #                            skel_data_init=skel_data_init, 
        #                            force_recompute=args.force_recompute)
        
        # trans = np.array(data['global_position']).reshape(1, 3)
        # rot = np.array(data['global_orient']).reshape(1, 3)
        # betas = np.array(data['betas']).reshape(1, 10)
        # body_pose = np.array(data['body_pose_axis_angle'])[0][:69].reshape(1, 69)        
        # body_pose = np.concatenate( (np.zeros((1, 3)), body_pose), axis=1 )
        
        
        trans=np.zeros((1, 3))
        rot=np.zeros((1, 3))
        betas=np.zeros((1, 10))
        body_pose=np.zeros((1, 72))
        
        
        skel_seq = skel_fitter.run_fit(trans,
                                       rot,
                                        betas,
                                        body_pose,
                                        batch_size=1,
                                        skel_data_init=skel_data_init, 
                                        force_recompute=args.force_recompute)

        print('Saved aligned SKEL to {}'.format(subj_dir))

        SKEL_skin_mesh = trimesh.Trimesh(vertices=skel_seq['skin_v'][0], faces=skel_seq['skin_f'])
        SKEL_skel_mesh = trimesh.Trimesh(vertices=skel_seq['skel_v'][0], faces=skel_seq['skel_f'])
        SMPL_mesh = trimesh.Trimesh(vertices=skel_seq['smpl_v'][0], faces=skel_seq['smpl_f'])
        
        # show the meshes
        # SKEL_skel_mesh.show()

        SKEL_skin_mesh.export(os.path.join(subj_dir, subj_name + '_skin.obj'))
        SKEL_skel_mesh.export(os.path.join(subj_dir, subj_name + '_skel.obj'))
        # SMPL_mesh.export(os.path.join(subj_dir, subj_name + '_smpl.obj'))
        send_skel(SKEL_skel_mesh.export(file_type="obj"))

        pickle.dump(skel_seq, open(pkl_path, 'wb'))
        exit(0)
    