# INBD
Iterative Next Boundary Detection for  Instance Segmentation of Tree Rings in Microscopy Images of Shrub Cross Sections

CVPR 2023. [Paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Gillert_Iterative_Next_Boundary_Detection_for_Instance_Segmentation_of_Tree_Rings_CVPR_2023_paper.html)

***

<img src="assets/example0.jpg" alt="Example input image and detected tree rings"/>

***

## Setup:

Python version: `3.7`. Other versions are known to cause issues.

```bash
#setup virtualenv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#download dataset
python fetch_dataset.py

#download pretrained models
python fetch_pretrained_models.py
```

Or use GitHub Codespaces: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=dev&repo=574937325&machine=standardLinux32gb&location=WestEurope)

***



## Inference

```bash
#single imagefile
python main.py inference checkpoints/INBD_EH/model.pt.zip dataset/EH/inputimages/EH_0033.jpg

#list of imagefiles
python main.py inference checkpoints/INBD_EH/model.pt.zip dataset/EH/test_inputimages.txt
```

### adding pith pixel position as argument
In Pinus taeda L. species, U-Net model do not segment the pith pixel correctly. In such species, pith can be model
as a single pixel ([see paper](https://arxiv.org/abs/2404.01952)). 
```bash

#single imagefile
python main.py inference checkpoints/UruDendro/model.pt.zip dataset/UruDendro4_1504/T0_B1_N32_A.png --cy 766 --cx 709
```
Where `--cy` and `--cx` are the y and x coordinates of the pith pixel in the image. Model checkpoints/UruDendro/model.pt.zip
was trained on the UruDendro dataset, which is a dataset of Pinus taeda L. tree rings ([see dataset](https://doi.org/10.5281/zenodo.15110646)).
Image `dataset/UruDendro4_1504/T0_B1_N32_A.png` is an example image from the UruDendro4 dataset ([see dataset](https://doi.org/10.5281/zenodo.15653340)).
resized to 1504x1504 pixels. Image can be downloaded from [link](https://finguy-my.sharepoint.com/:i:/g/personal/henry_marichal_fing_edu_uy/Efpex2iHvplKt3c1NDdK7CUBYCiF-5VOi8N3XV51rEPQiw?e=rF9trA).
***

## Training:


```bash
#first, train the 3-class segmentation model
python main.py train segmentation           \
  dataset/EH/train_inputimages.txt          \
  dataset/EH/train_annotations.txt

#next, train the inbd network
python main.py train INBD \
  dataset/EH/train_inputimages.txt          \
  dataset/EH/train_annotations.txt          \
  --segmentationmodel=checkpoints/segmentationmodel/model.pt.zip   #adjust path
```



***

## Dataset

We introduce a new publicly available dataset: MiSCS (Microscopic Shrub Cross Sections)

The dataset and annotations can be downloaded via `python fetch_dataset.py` or via the following links:
- [DO (Dryas octopetala)](https://github.com/alexander-g/INBD/releases/download/dataset_v2/DO_v2.zip)
- [EH (Empetrum hermaphroditum)](https://github.com/alexander-g/INBD/releases/download/dataset_v2/EH_v2.zip)
- [VM (Vaccinium myrtillus)](https://github.com/alexander-g/INBD/releases/download/dataset_v2/VM_v2.zip)

All images were acquired by Alba Anadon-Rosell.
If you have ecology-related questions, please contact `a.anadon at creaf.uab.cat`

If you want to use this dataset for computer vision research, please cite the publication as below.


## Citation

```bibtex
@inproceedings{INBD,
  title     = "{Iterative Next Boundary Detection for Instance Segmentation of Tree Rings in Microscopy Images of Shrub Cross Sections}",
  author    = {Alexander Gillert and Giulia Resente and Alba Anadon‐Rosell and Martin Wilmking and Uwe von Lukas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {14540-14548}
}
```

***

## License

License for the source code: [MPL-2.0](https://github.com/alexander-g/INBD/blob/master/LICENSE)

License for the dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)



