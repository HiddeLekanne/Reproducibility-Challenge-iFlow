# Reproducibility challenge PapersWithCode: iFlow, ICLR2020.

This repository is the a re-implementation of [Identifying through Flows for Recovering Latent Representations](https://arxiv.org/abs/1909.12555). 
With source code used from the [official repository](https://github.com/MathsXDC/iFlow).

## Requirements
This repository uses anaconda for environment management and pytorch for machine learning.

To install and use the cpu environment used in our experiments do:
```
conda env create -f environment.yml
conda activate iFlow
```
For precise reproducability a nvidia 1080 Ti is needed with cuda version 10.1 and cudnn version 7.6
To install and use the cuda environement used in our experiments do:
```
conda env create -f environment-cuda.yml
conda activate iFlow-cuda
```

## Training
TODO: fix the scrips mentioned here
To train a iFlow model, run this command from the iFlow directory:

```train
cd iFlow
./scripts/run_iFlow.sh
```

A more comprehensive overview is given in the jupyter notebook completeRun. Here the exact configurations used in our reproducibility paper.
```completerun
cd iFlow
jupyter notebook
```
## Evaluation

All plots used in the paper can be reprodced and configured in the notebook completeRun.

## Pre-trained Models
TODO
You can download pretrained models here:

<!---- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. --->

<!---- >📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. --->

## Results
TODO
Our model achieves the following performance on :

<!----### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

<!----| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
<!----| ------------------ |---------------- | -------------- |
<!----| My awesome model   |     85%         |      95%       |

<!---- >📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


<!----## Contributing

<!---- >📋  Pick a licence and describe how to contribute to your code repository. 
