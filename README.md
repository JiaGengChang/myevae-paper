# Cancer Survival ML

Using machine learning to predict survival in a right-censored, high dimensional NGS dataset of cancer patients. 

<img src="./assets/cancer-ml-logo-export.svg" alt="Multi-omics risk modelling logo">


## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

Provide a brief introduction to the project and its objectives.

## Dataset

We used the CoMMpass (Relating Clinical Outcomes in MM to Personal Assessment of Genetic Profile) trial (NCT01454297) data of Multiple Myeloma Research Foundation (MMRF) [[1](#references)]. 

We used two different versions of CoMMpass data, Interim Analysis 16 (IA16) and Interim Analysis 21 (IA21) versions. This is because IA16 is the latest release which has information on IGH translocation partners. 

## Models

### Model architecture
The layers and layer dimensions of VAE risk model is as shown:

<img src="./assets/vae-diagram-export.svg" alt="Using variational autoencoder to integrate omics data">

1. Data from Whole genome sequencing (WGS), whole exome sequencing (WXS), and RNA-Sequencing (RNA-Seq) are first individually encoded using the peripheral encoder layers.
    1. WGS IA21*: Gene level copy number, GISTIC recurrently amplified/deleted regions, interphase FISH probe locations
    2. WXS IA21: Single Base Substitution Mutational signatures
    3. RNA-Seq IA21: Gene level transcripts per million
    4. WGS IA16/RNA-Seq IA16: IgH translocation partner classification
       
2. Encoded data is concatenated and jointly encoded by passing through the bottleneck layer.
   
3. For risk prediction, bottleneck embeddings (`z`) are concatenated with clinical information and passed through a fully connected layer.
    1. clinical information: age, sex, ISS stage, retrieved from IA21
    2. desired values: right-censored progression-free survival (PFS) or overall survival (OS), retrieved from IA21
       
4. For input reconstruction, bottleneck embeddings (`z`) are passed through the bottleneck decoder layer and peripheral decoder layers.

### Training

We use 10-random samples of 5-fold cross validation to tune hyperparameters.

Hyperparameters we tuned include 
* bottleneck layer dimension: z=2,4,8,16,32
* activation/non-linearity function: ReLU, Sigmoid, LeakyReLU, Tanh (for encoder, decoder, and task networks)
* learning rate: 1e-5, 5e-5, 1e-4, 3e-4, 1e-3
* KL divergence loss weight: 0.1, 0.5, 1, 2, 4
* input preprocessing: min-max scaling, standardization, tanh, arcsinh

Our desired value to model is progression free survival. The best epoch is the one with lowest validation survival loss (not the metric). Early stopping based on validation survival loss with a patience of 20 is used. A burn-in of 50 epochs is used.

After training, PDF files of convergence plots will be produced in the output folder. This monitors the KL divergence loss, reconstruction loss for every data modality, survival loss (negative log likelihood; NPLL), and C-index metric on validation dataset.

An example of training convergence is shown here. Red is for the validation dataset, blue for training.

<img src="./assets/example-pfs-shuffle-0-fold-0.png" alt="Loss curves for training and validation datasets">

The drop in survival loss (and improvement in metric) typically coincides with a spike KL divergence. This indicates that the latent distribution is deviating away from the prior N_z(0,1). 

However, as our main task is to improve survival loss rather than to use the latent embeddings for generative modelling, we accept this increase in KL divergence as a sacrifice to improving on the survial modelling task.

### Validation

In this section, we describe the procedure to evaluate multiple models.

Run `make eval` or `python modules_vae/eval.py` to extract bottom-line validation metric for all models in the output directory. 

It requires as input the result json files for every model. An example is `output/example/pfs_shuffle0_fold0.json`. Each time a train script is submitted, 50 of these .json files will be produced.

This script creates a results file called `model_scores.json` which can be used for model comparison. It calculates the mean validation metric across the 50 models and its 95% confidence interval.

## Results

Summarize the results obtained from the models. Include any relevant metrics and visualizations.

## References

1. [Multiple Myeloma Research Foundation (MMRF)](https://themmrf.org/)

## Acknowledgements

Our gratitude goes to the data provider, [Multiple Myeloma Research Foundation (MMRF)](https://themmrf.org/) CoMMpass (Relating Clinical Outcomes in MM to Personal Assessment of Genetic Profile) trial (NCT01454297)

Data is available on https://research.themmrf.org but one must email them for permissions first.

## License

This project is licensed under the GNU General Public License v3.0. You may copy, distribute, and modify the software as long as you track changes/dates in source files. Any modifications to or software including (via compiler) GPL-licensed code must also be made available under the GPL along with build & install instructions.
