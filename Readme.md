# Code & Data for GDGCN
This is an implementation of GDGCN. Following this code, you can easily reproduce GDGCN and conduct detailed further ablation experiments. This link is anonymous currently and will be made public if the paper is accepted. At that time, more detailed questions about this work will be welcomed. 

# Environment
- Python 3.6.12
- PyTorch 1.6.0
- NumPy 1.19.1
- tqdm 4.51.0

# Dataset
The datasets used in our paper are collected by the Caltrans Performance Measurement System(PeMS). Please refer to [STSGCN](https://github.com/Davidham3/STSGCN) (AAAI2020) for the original download url. Original data can be processed into input data by generating_pems_training_data.py. The preprocessed data of PEMS08 are placed in ./data/PEMS08, which can be downloaded and directly used as input.

# Training 
- Train the PEMS03: 
    ```python train.py --num_nodes=358 --data=./data/PEMS03 --layers=8 --nhid=16```
- Train the PEMS04:
    ```python train.py --num_nodes=307 --data=./data/PEMS04 --layers=8 --nhid=64``` 
- Train the PEMS07: 
    ```python train.py --num_nodes=883 --data=./data/PEMS07 --layers=5 --nhid=64```
- Train the PEMS08: 
    ```python train.py --num_nodes=170 --data=./data/PEMS08 --layers=6 --nhid=16```
  
We give the examples using default parameter settings above. More hyperparameters can be viewed in the **source code** and the submitted article **Appendix**, adjusted with the running demand.

# More Experimental Details
  We further introduce how to conduct experiments to replace the Temporal Graph Convolutional Block. In the article, the experimental results of this part is shown in the **Table 6** of the submitted paper.

  - Run **GDGCN w/o temproal**: ```python train.py --temporal_mode=no_temporal```. It refers that we only learn spatial information in this module and temporal graph convolution block is removed.
  
  - Run **GDGCN w Attention**: ```python train.py --temporal_mode=attention```. It refers that temporal graph convolution block is replaced by attention.
  
  - Run **GDGCN w TCN**: ```python train.py --temporal_mode=tcn```. It refers that temporal graph convolution block is replaced by TCN.
  
  - Run **GDGCN w LSTM**: ```python train.py --temporal_mode=lstm```. It refers that temporal graph convolution block is replaced by LSTM.
  
