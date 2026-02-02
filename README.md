# Towards Safe Navigation Through Crowded Dynamic Environments
Training code for the CNN control policy proposed in our paper ["Towards Safe Navigation Through Crowded Dynamic Environments"](https://doi.org/10.1109/IROS51168.2021.9636102) ([online](https://sites.temple.edu/trail/files/2021/11/XieXinDamesIROS2021.pdf)), published in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 

## Requirements:
* Pytorch 1.7.1

## Usage:
Assuming you have already collected the dataset and placed it in your home directory.
A potential dataset can be used for training is our [Semantic2D dataset](https://doi.org/10.5281/zenodo.13730200).
```
git clone https://github.com/TempleRAIL/cnn_nav.git
# training:
cd cnn_nav
sh run_train_eval.sh ~/dataset/train ~/dataset/dev 
# evaluation:
cd cnn_nav
sh run_eval.sh ~/dataset/test 
```


## Citation
```
@inproceedings{xie2021towards,
  title={Towards safe navigation through crowded dynamic environments},
  author={Xie, Zhanteng and Xin, Pujie and Dames, Philip},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4934--4940},
  year={2021},
  organization={IEEE}
}
```
