# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Requirements
This project requires a working Python3 environment with numpy and pytorch intalled. Image data is not part of this repository.

## Training
To train the network, run

```
python train.y <data_dir>
```

For futher options see the console output. The training result will be saved to a checkpoint file.

## Predictions
To predict the category of an image use
```
python predict.py <image_file> <checkpoint>
```

For further options see the console output.


