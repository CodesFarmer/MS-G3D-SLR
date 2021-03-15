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

- Resume training from checkpoint
```
python3 -u -m torch.distributed.launch --nproc_per_node=8 main_dist \
  --config <config file> \
  --work-dir <place to keep things (weights, checkpoints, logs)> \
  --checkpoint <checkpoint to be resumed> \
  [--base-lr <base learning rate>] \
  [--batch-size <batch size>] \
  [--weight-decay <weight decay>] \
  [--forward-batch-size <batch size during forward pass, useful if using only 1 GPU>]
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

