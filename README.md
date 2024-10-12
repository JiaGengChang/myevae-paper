# Cancer Survival ML

Using machine learning to predict survival in a right-censored, high dimensional NGS dataset of cancer patients. 

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

<img src="./vae-diagram-export.svg" alt="Using variational autoencoder to integrate omics data">

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

### Validation

## Results

Summarize the results obtained from the models. Include any relevant metrics and visualizations.

## References

1. [Multiple Myeloma Research Foundation (MMRF)](https://themmrf.org/)

## Acknowledgements

Our gratitude goes to the data provider, [Multiple Myeloma Research Foundation (MMRF)](https://themmrf.org/) CoMMpass (Relating Clinical Outcomes in MM to Personal Assessment of Genetic Profile) trial (NCT01454297)

Data is available on https://research.themmrf.org but one must email them for permissions first.

## License

This project is licensed under the GNU General Public License v3.0. You may copy, distribute, and modify the software as long as you track changes/dates in source files. Any modifications to or software including (via compiler) GPL-licensed code must also be made available under the GPL along with build & install instructions.
