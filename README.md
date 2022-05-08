# Vision transformer for feature extraction in Abstract visual reasoning task. 

In this repository we implement a Vision Transformer based Relational network to solve Raven's progressive matrices. We use I-RAVEN dataset for this task, and we also compare  this novel approach with previously applied model architectures. 

# Dependencies

**Important**
*numpy
*scipy==1.1.0
*scikit-image
*seaborn
*torch
*torchvision
*tqdm
*future
*tensorboard
*tensorboardX

See ```requirements.txt``` for other dependencies.

Run
```Bash
python main.py --model <WReN/CNN_MLP/Resnet50_MLP/LSTM/ViT_WReN/ViT_LSTM/ViT_MLP> --img_size <input image size> --path <path to your dataset>
```

# Acknowledgement
- https://github.com/husheng12345/SRAN
- https://github.com/Fen9/WReN
