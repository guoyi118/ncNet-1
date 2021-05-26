# ncNet
Code for ncNet

## Environment Setup

* `Python3.6`
* `Pytorch 1.8.1` or higher

Install Python dependency via `pip install -r requirements.txt` when the environment of Python and Pytorch is setup.


## Running Code

#### Data preparation

<!-- * Download [Glove Embedding](xxxxx) and put `glove.6B.100d` under `./dataset/` directory -->

* [Must] Download the Spider data [here](https://drive.google.com/drive/folders/1wmJTcC9R6ah0jBo_ONaZW3ykx5iGMx9j?usp=sharing) and unzip under `./dataset/` directory

* [Optional] **_Only if_** you change the `train/dev/test.csv` under the `./dataset/` folder, you need to run `process_dataset.py` under the `preprocessing` foler. 

#### Runing Example

Open the `test_ncNet.ipynb` to try the running example.

#### Training

Run `train.py` to train ncNet.


#### Testing

Run `test.py` to eval ncNet.
