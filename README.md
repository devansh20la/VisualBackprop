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

![n02169497_1092_img](https://user-images.githubusercontent.com/16810812/52000813-03d7b380-248c-11e9-9d4b-329dfc11a24f.png)
![n02169497_1092_outputseg](https://user-images.githubusercontent.com/16810812/52000814-03d7b380-248c-11e9-8c03-3ee08155b44f.png)
![n02169497_1092_over](https://user-images.githubusercontent.com/16810812/52000815-03d7b380-248c-11e9-81ad-cc6e171d6cfe.png)
![n02169497_1122_img](https://user-images.githubusercontent.com/16810812/52000816-03d7b380-248c-11e9-9095-4e9aa5108b6d.png)
![n02169497_1122_outputseg](https://user-images.githubusercontent.com/16810812/52000817-03d7b380-248c-11e9-96f4-a4e799a0ca23.png)
![n02169497_1122_over](https://user-images.githubusercontent.com/16810812/52000818-03d7b380-248c-11e9-89c9-4c14d97cdfca.png)
![n02169497_1307_img](https://user-images.githubusercontent.com/16810812/52000819-03d7b380-248c-11e9-8259-dfc98e4cdb1b.png)
![n02169497_1307_outputseg](https://user-images.githubusercontent.com/16810812/52000820-03d7b380-248c-11e9-825a-c7ad18398b6a.png)
![n02169497_1307_over](https://user-images.githubusercontent.com/16810812/52000821-03d7b380-248c-11e9-8e28-3bcca6374866.png)
![n02169497_1309_img](https://user-images.githubusercontent.com/16810812/52000822-03d7b380-248c-11e9-9996-8f3de18dc94c.png)
![n02169497_1309_outputseg](https://user-images.githubusercontent.com/16810812/52000823-04704a00-248c-11e9-8bf8-452e7fce5ab0.png)
![n02169497_1309_over](https://user-images.githubusercontent.com/16810812/52000824-04704a00-248c-11e9-9330-87075644ee74.png)

## Acknowledgments

We thank Dr. Marius Bojarski NVIDIA Corporation for inpirations and useful feedbacks.



