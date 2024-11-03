# LabelAugmentation

## Requirements

```bash
conda create -n LabelAugmentation python=3.11
conda activate LabelAugmentation
```

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c anaconda jupyter numpy
conda install -c conda-forge tqdm torchmetrics wandb matplotlib

pip install git+https://github.com/f-amerehi/Distort-Images.git
pip install imagecorruptions
pip install cleverhans
```
