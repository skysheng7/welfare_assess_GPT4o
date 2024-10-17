# Cow welfare assessment using ChatGPT Vision

This repository contains the code for our project: **Cow welfare assessment using ChatGPT Vision**.

## Getting Started
1. Repository Structure
Here's a brief overview of the repository's structure:

- **results_video_quality**: Contains results for using GPT-4o for video quality check 
- **results_welfare_assess**: Contains results for using GPT-4o for welfare assessment
- **venv**: Provides details about the virtual environment for python.
- **src**: Contains scripts used for connecting to GPT-4o, prompting it for video quality and welfare assessment.

Thank you for your interest in our project. We hope you find the code insightful!

## Dataset Information

- **Title of Dataset:** Replication Data for: Cow welfare assessment using ChatGPT Vision
- **Conference Presented at:** the 9th International Conference on the Welfare Assessment of Animals at Farm Level (WAFL)
- **Dataset Created:** 2024-02-01
- **Created by:** Kehan (Sky) Sheng
- **Contact Email:** <skysheng7@gmail.com>

## Abstract published in WAFL
Animal-based welfare assessment is usually conducted by trained assessors. Farm audits typically cover only a fraction of the herd, and the results vary based on the assessor's skill and experience. The infrequent, costly, and time-consuming nature of farm audits underlines the potential advantages of adopting automated assessment methods. We employed a large multi-modal model, chatGPT Vision (GPT-4V), to automate the binary classification of cattle cleanliness (i.e., dirty and clean). We trained GPT-4V with the Welfare Quality Protocol (WQP) text instructions and provided 2 image examples per category to categorize dairy cow cleanliness across 3 body parts: hind leg, hindquarter, and udder. Our test dataset included 24 images (4 dirty and 4 clean images per body part) that have been previously assessed by 5 experts and were used in auditor training. To improve model performance, we used prompt engineering techniques including role-playing (i.e., “act as an animal welfare assessor with 20 years of experience”) and emotion prompting (i.e., “this is vital for my career”). GPT-4V demonstrated an acceptable level of agreement with expert assessments across the entire dataset (Cohen’s Kappa=0.42, P=0.01), but was biased towards labeling images as dirty (precision = 0.63, recall=1.00). It particularly excelled in the evaluation of lower hind leg cleanliness (Cohen’s Kappa=0.75, P=0.03; precision=0.8, recall=1.00). Considering only the cleanliness of lower hind leg and hindquarter areas, the agreement reached a level which is considered acceptable for welfare quality control organizations (Cohen’s Kappa=0.63, P=0.01; precision=0.73, recall=1.00). To eliminate background noise and potentially boost model performance, we applied two image processing techniques: segmenting the cow from the background and isolating the target body part for assessment. Both techniques did not yield improved results. We recommend future researchers to provide more training examples to GPT-4V to enhance model performance and extensive testing of large multi-models across various WQP criteria to further assess the feasibility of automated welfare assessments on farms.

- **Note**: The abstract included results generated using GPT-4V when we submited the abstract to WAFL. However, we updated our results using GPT-4o in August, 2024, which are included in the poster and in the results folder in this repository.

## Contributors

- **Principal Investigator:** Borbala Foris  
  - ORCID: 0000-0002-0901-3057  
  - Affiliation: University of Veterinary Medicine, Vienna
  - Email: <forisbori@gmail.com>

- **Contributor:** Kehan Sheng  
  - ORCID: 0000-0001-6442-5284  
  - Affiliation: University of British Columbia  
  - Email: <skysheng7@gmail.com>

- **Contributor:** Mahshid Heydarirad  
  - Affiliation: University of British Columbia  
  - Email: <mahshid.heydarirad@yahoo.com>

