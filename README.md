# Stratified Rule-Aware Network for Abstract Visual Reasoning
This repository contains implementation of our AAAI 2021 paper.

[Stratified Rule-Aware Network for Abstract Visual Reasoning](https://arxiv.org/abs/2002.06838)  
Sheng Hu\*, Yuqing Ma\*, Xianglong Liu†, Yanlu Wei, Shihao Bai  
*Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, 2021  
(\* equal contribution, † corresponding author)

# I-RAVEN Dataset
To fix the defacts of RAVEN dataset, we generate an alternative answer set for each RPM question in RAVEN, forming an improved dataset named Impartial-RAVEN (I-RAVEN for short). The comparison between the two datasets is shown below. For more details, please refer to our paper.  
<div  align="center">    
<img src="https://raw.githubusercontent.com/husheng12345/SRAN/master/Images/I-RAVEN.png" width="70%">
</div>  

## Dataset Generation
Code to generate the dataset resides in the ```I-RAVEN``` folder. The dependencies are consistent with [the original RAVEN](https://github.com/WellyZhang/RAVEN).
* Python 2.7
* OpenCV
* numpy
* tqdm 
* scipy
* pillow

See ```I-RAVEN/requirements.txt``` for a full list of packages required. To install the dependencies, run
```
pip install -r I-RAVEN/requirements.txt
```
To generate a dataset, run
```
python I-RAVEN/main.py --num-samples <number of samples per configuration> --save-dir <directory to save the dataset>
```
Check the ```I-RAVEN/main.py``` file for a full list of arguments you can adjust.

# Stratified Rule-Aware Network 
<div  align="center">  
<img src="https://raw.githubusercontent.com/husheng12345/SRAN/master/Images/SRAN.png" width="80%">
</div>  

Code of our model resides in the ```SRAN``` folder. The requirements are listed as follows:

* Python 3.7
* CUDA
* PyTorch
* torchvision
* scipy 1.1.0
* Visdom

See ```SRAN/requirements.txt``` for a full list of packages required. To install the dependencies, run
```
pip install -r SRAN/requirements.txt
```

To view training results, run ```python -m visdom.server -p 9527``` and click the URL [http://localhost:9527](http://localhost:9527).


To train and evaluate the model, run
```
python SRAN/main.py --dataset <I-RAVEN or PGM> --dataset_path <path to the dataset> --save <directory to save the checkpoint>
```

Check the ```SRAN/main.py``` file for a full list of arguments you can adjust.

# Performance
Performance on I-RAVEN:

| Model      | Acc        | Center     | 2x2G	    | 3x3G	     | O-IC       | O-IG       | L-R        | U-D        |
| :---       | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      | :---:      |
| LSTM	| 18.9%  	  | 26.2%      | 16.7%      | 15.1%      | 21.9%      | 21.1%      | 14.6%      | 16.5%      |
| WReN [[code](https://github.com/Fen9/WReN)]       | 23.8%      | 29.4%      | 26.8%      | 23.5%      | 22.5%      | 21.5%      | 21.9%      | 21.4%      |
| ResNet	| 40.3%      | 44.7%      | 29.3%      | 27.9%      | 46.2%      | 35.8%      | 51.2%      | 47.4%      |
| ResNet+DRT [[code](https://github.com/WellyZhang/RAVEN/blob/master/src/model/resnet18.py)]	| 40.4%      | 46.5%      | 28.8%      | 27.3%      | 46.0%      | 34.2%      | 50.1%      | 49.8%      |
| LEN [[code](https://github.com/zkcys001/distracting_feature)]	| 41.4%      | 56.4%      | 31.7%      | 29.7%      | 52.1%      | 31.7%      | 44.2%      | 44.2%      |
| Wild ResNet| 44.3%      | 50.9%      | 33.1%      | 30.8%      | 50.9%      | 38.7%      | 53.1%      | 52.6%      |
| CoPINet [[code](https://github.com/WellyZhang/CoPINet)]	| 46.1%      | 54.4%      | 36.8%      | 31.9%      | 52.2%      | 42.8%      | 51.9%      | 52.5%      |
| SRAN (Ours)| **60.8%**  | **78.2%**  | **50.1%**  | **42.4%**  | **68.2%**  | **46.3%**  | **70.1%**  | **70.3%**  |

Performance on PGM:

| Model  | LSTM    | ResNet | Wild ResNet | CoPINet    | WReN 	| MXGNet 	| LEN 	| SRAN (Ours)	|
| :---   | :---:   | :---:  | :---:       | :---:      | :---:	| :---:		| :---:	| :---: 		|
| Acc    | 35.8%   | 42.0% 	| 48.0%       | 56.4%      | 62.6%	| 66.7%		| 68.1%	| **71.3%** 	|

# Citation
If you find our work helpful, please cite us.
```
@inproceedings{hu2021stratified,
     title={Stratified Rule-Aware Network for Abstract Visual Reasoning},
     author={Hu, Sheng and Ma, Yuqing and Liu, Xianglong and Wei, Yanlu and Bai, Shihao},
     booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
     volume={35},
     number={2},
     pages={1567--1574},
     year={2021}
 }  
```




