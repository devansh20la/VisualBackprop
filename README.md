# VisualBackprop
This is the Pytorch version of the VisualBack prop algorithm on the VGG16 model. Theoretical description of the algorithm can be found found here: https://arxiv.org/abs/1611.05418.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

```
Pytorch 1.0
```

## Run

The input data should be arranged in standard pytorch data loading format i.e
```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

```
python main.py --cp='trained model' --bs='Batch Size' --ms='Manual Seed' --dir='Data Directory'
```

## Results
The results are saved in the results directory. Here are sample results for VGG16 model pretrained on ImageNet. 


## Acknowledgments

We thank Dr. Marius Bojarski NVIDIA Corporation for inpirations and useful feedbacks.



