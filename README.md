
# Pavement crack detection: dataset and model
The project is used to share our recent work on pavement crack detection. For the details of the work, the readers are refer to the paper "Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection" (FPHB), T-ITS 2019.
You can find the paper in https://www.researchgate.net/publication/330244656_Feature_Pyramid_and_Hierarchical_Boosting_Network_for_Pavement_Crack_Detection or https://arxiv.org/abs/1901.06340.

The pavement crack datasets used in paper, crack detection results on each datasets, trained model, and crack annotation tool are stored in [Google Drive](https://drive.google.com/open?id=1y9SxmmFVh0xdQR-wdchUmnScuWMJ5_O-) and [Daidu Yunpan](https://pan.baidu.com/s/1JwJO96BOtJ50MykBcYKknQ) extract code: jviq.
**If you think this project is useful for you, feel free to leave a star. (^^)**
# Installing
1. Install prerequisites for Caffe
2. Clone the repository 
```shell
git clone https://github.com/fyangneil/pavement-crack-detection.git
```
3. Build Caffe
```shell
cd $ROOT_DIR/pavement-crack-detection
make -j8&make pycaffe
```

# Training
The training and test steps are same with HED, please read the instruction in https://github.com/s9xie/hed.
Here we use CRACK500 dataset as an example to demonstrate how to set experiment (assume you have successfully train and test HED on BSD500 dataset).


1. Create a "crack" folder in "pavement-crack-detection/data' folder.
```shell
cd $ROOT_DIR/pavement-crack-detection/data
mkdir crack
```
2. Download CRACK500 and extract it to crack folder and put "train.txt" in "crack" folder. 
3. Download the fully convolutional VGG model (248MB) from [here](http://vcl.ucsd.edu/hed/5stage-vgg.caffemodel) and put it in "pavement-crack-detection/examples/fphb" folder.
4. Train fpn and fphn model on CRACK500 data 
```shell
cd $ROOT_DIR/pavement-crack-detection/examples/fphb
python solve_fphb_crack.py or python solve_fpn_crack.py
  ```

# Test
We use CRACk500 as an example to demonstrate the process of testing the trained model.
1. Download CRACK500 and extract it to "crack" folder and put "test.txt" in "crack" folder.  
2. Test the model
```shell
cd $ROOT_DIR/pavement-crack-detection/examples/fphb
python test.py
```
When testing on one image, you are referred to "pavement-crack-detection/examples/fphb/FPHB-tutorial.ipynb" file.
# Evaluate result
We provide the evaluation tool "eval_tool" to assess the result. Before using the tool, please make sure that in predicted crack map the bright regions are crack, background is black. To get ODS and OIS, run 'crack_nms.m' first then 'crack_eval.m'. To get AIU, run 'crack_AIU.m'.

**If you encounter any issue when using our code or model, feel free to contact me fyang@temple.edu.**

# Note: please cite the corresponding papers when using these datasets.
CRACK500:
>@inproceedings{zhang2016road,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={Image Processing (ICIP), 2016 IEEE International Conference on},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}' .

>@article{yang2019feature,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}

GAPs384: 
>@inproceedings{eisenbach2017how,
  title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.},
  author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus
          and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike
          and Gross, Horst-Michael},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  pages={2039--2047},
  year={2017}
}

>@article{yang2019feature,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}

CFD: 
>@article{shi2016automatic,
  title={Automatic road crack detection using random structured forests},
  author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={12},
  pages={3434--3445},
  year={2016},
  publisher={IEEE}
}

AEL: 
>@article{amhaz2016automatic,
  title={Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection.},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent}
}

cracktree200: 
>@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}


