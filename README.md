# Towards Explainable Sign Spotting Systems: an Exploration of Approximative Linguistic Features and Evaluation Methods
## Data Science Master thesis by Natalie Hollain
### Radboud University (2022-2023)

### Published papers
- [Distractor-based Evaluation of Sign Spotting](https://github.com/nataliehh/Evaluating-Sign-Spotting)
- [Analyzing the Potential of Linguistic Features for Sign Spotting: A Look at Approximative Features](https://github.com/nataliehh/Linguistic-Features-for-Sign-Spotting)

---

## Setup 
** NOTE: the instructions below may no longer be up to date! **

- Please use `requirements.txt` to install the dependencies for the code. 
- Create the CNGT_data folder, which will be used to store the required files you need to run the model. 
- Note that the linguistic data and video data are not provided because they were privately shared with us. Obtain the data from [Corpus NGT](https://www.corpusngt.nl/) and [NGT Signbank](https://signbank.cls.ru.nl/). Please contact the authors of these data to gain access. The data we use are:
	1. Corpus NGT videos and annotation files in the format `Sxxx_CNGTyyyy.mpg`, `Sxxx_CNGTyyyy.eaf`
	2. All gloss information and the minimal pairs from NGT Signbank

- We first extract landmarks from Corpus NGT. Run `mediapipe_processing.ipynb` (remove the break in the for-loop to process all files). We warn that this takes several days to run even when done in parallel. 
- Now, we can execute the `data_demographics_split.ipynb` notebook to obtain the split of the signers into the train, validation and test data.
- Run `gloss_selection.ipynb` to obtain which signs are to be used for the dataset creation.
- Execute `linguistic_distance.df` to obtain the linguistic distances between the signs.
- Next, use `sign_spotting_dataset_creation.ipynb` to create the dataset for the training of our sign spotting model. The feature extraction from the Mediapipe landmarks is done here, and it takes some time (30-60 minutes). The analysis of the correlation takes even longer (at least an hour). 
- Run `sign_spotting_train.ipynb` to train the model.  **Note: please run the notebook thrice**, once for each set of features. You need to change the parameter `model_mode` each time (change the index at the end).
- Finally, execute the code in `sign_spotting_analysis.ipynb` to get the performance of each model.