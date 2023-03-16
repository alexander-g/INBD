# INBD
Iterative Next Boundary Detection for  Instance Segmentation of Tree Rings in Microscopy Images of Shrub Cross Sections

Accepted to CVPR 2023. Preprint: [https://arxiv.org/pdf/2212.03022.pdf](https://arxiv.org/pdf/2212.03022.pdf)

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

Or use GitHub Codespaces: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=dev&repo=574937325&machine=basicLinux32gb&location=WestEurope)

***



## Inference

```bash
#single imagefile
python main.py inference checkpoints/INBD_empetrum/model.pt.zip imagefile.JPG

#list of imagefiles
python main.py inference checkpoints/INBD_empetrum/model.pt.zip dataset/EH/test_inputimages.txt
```

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
- [DO (Dryas octopetala)](https://github.com/alexander-g/INBD/releases/download/dataset_v1/DO_v1.zip)
- [EH (Empetrum hermaphroditum)](https://github.com/alexander-g/INBD/releases/download/dataset_v1/EH_v1.zip)
- [VM (Vaccinium myrtillus)](https://github.com/alexander-g/INBD/releases/download/dataset_v1/VM_v1.zip)

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
  note      = {Preprint: \url{https://arxiv.org/pdf/2212.03022.pdf}}
}
```

***

## License

License for the source code: [MPL-2.0](https://github.com/alexander-g/INBD/blob/master/LICENSE)

License for the dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)



