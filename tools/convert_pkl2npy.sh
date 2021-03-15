#!/bin/bash

# python convert_pkl2npy.py \
#   --data /workspace/dataset/AUTSL/frames_bdvis/autsl_skeleton_joints_train.pkl \
#   --output autsl_skeleton_joints_train_t64.npy \
#   --label train_label.pkl \
#   --nclips 64
# 
# 
# python convert_pkl2npy.py \
#   --data /workspace/dataset/AUTSL/frames_bdvis/autsl_skeleton_joints_val.pkl \
#   --output autsl_skeleton_joints_val_t64.npy \
#   --label val_label.pkl \
#   --nclips 64
# 
# 

# python convert_pkl2npy.py \
#   --data autsl_skeleton_joints_smoothed_train.pkl \
#   --output autsl_skeleton_joints_smoothed_train_t64.npy \
#   --label train_label.pkl \
#   --nclips 64
# 
# 
# python convert_pkl2npy.py \
#   --data autsl_skeleton_joints_smoothed_val.pkl \
#   --output autsl_skeleton_joints_smoothed_val_t64.npy \
#   --label val_label.pkl \
#   --nclips 64


python convert_pkl2npy.py \
  --data autsl_skeleton_joints_bvphfv210307_flow_3d_train.pkl \
  --output minitrain_data_joint.npy \
  --label minitrain_label.pkl \
  --method middle \
  --nclips 96


python convert_pkl2npy.py \
  --data autsl_skeleton_joints_bvphfv210307_flow_3d_train.pkl \
  --output minival_data_joint.npy \
  --label minival_label.pkl \
  --method middle \
  --nclips 96
 
 

python convert_pkl2npy.py \
  --data autsl_skeleton_joints_bvphfv210307_flow_3d_val.pkl \
  --output val_data_joint.npy \
  --label val_label.pkl \
  --method middle \
  --nclips 96
 
 

