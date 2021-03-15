# MS-G3D distributed version
This is an variant version of [MS-G3D](https://github.com/kenziyuliu/MS-G3D)  
The difference is  
1. Distributed Training  
2. SE module
3. Complement Graph and Full Graph
4. Dense Padding

## Dependencies
- Python >= 3.7
- PyTorch >= 1.5.1
- [NVIDIA Apex](https://github.com/NVIDIA/apex) (auto mixed precision training)
- PyYAML, tqdm, tensorboardX


## Training & Testing

- The general training template command:
```
python3 -u -m torch.distributed.launch --nproc_per_node=8 main_dist \
  --config <config file> \
  --work-dir <place to keep things (weights, checkpoints, logs)> \
  [--base-lr <base learning rate>] \
  [--batch-size <batch size>] \
  [--weight-decay <weight decay>] \
  [--forward-batch-size <batch size during forward pass, useful if using only 1 GPU>]
```

- The general testing template command:
```
python3 -u main.py
  --config <config file>
  --work-dir <place to keep things>
  --device <GPU IDs to use>
  --weights <path to model weights>
  [--test-batch-size <...>]
```

- Template for joint-bone two-stream fusion:  
You can build a list of scores generated at test stage, and create a file list such as:  
ensemble.txt  
  path/to/first/model/test/score.pkl 1  
  path/to/second/model/test/score.pkl 1  
  ...  
  path/to/last/model/test/score.pkl 1  
```
cd ensemble  
python3 ensemble.py \
  --ground-truth <label_file> \  
  --method <softmax / sum> \  
  --output <output fiel to save merged score> \  
  --result-list <models list to be merged>  
```

- Use the corresponding config files from `./config` to train/test different datasets

- Examples
  - Train on NTU 120 XSub Joint
    - Train with 1 GPU:
      - `python3 main.py --config ./config/nturgbd120-cross-subject/train_joint.yaml`
    - Train with 2 GPUs:
      - `python3 main.py --config ./config/nturgbd120-cross-subject/train_joint.yaml --batch-size 32 --forward-batch-size 32 --device 0 1`
  - Test on NTU 120 XSet Bone
    - `python3 main.py --config ./config/nturgbd120-cross-setup/test_bone.yaml`
  - Batch size 32 on 1 GPU (BS 16 per forward pass by accumulating gradients):
    - `python3 main.py --config <...> --batch-size 32 --forward-batch-size 16 --device 0`

- Resume training from checkpoint
```
python3 main.py
  ...  # Same params as before
  --start-epoch <0 indexed epoch>
  --weights <weights in work_dir>
  --checkpoint <checkpoint in work_dir>
```


## Acknowledgements

This repo is based on
  - [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
  - [ST-GCN](https://github.com/yysijie/st-gcn)
  - [MS-G3D](https://github.com/kenziyuliu/MS-G3D)

Thanks to the original authors for their work!


```

## Contact
Please email `jhonjoe.c@gmail.com` for further questions

