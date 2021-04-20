## Introduction

This repo contains all the codes of topic modeling.

- topic_modeling.py : this is the principal file of our topic modeling using (NMF, LDA, BERT and LDA) 
- topic_modeling.ipynb : the notebook version of topic_modeling.py
- destinations.csv : this file stores the names of all countries (it is used for the analysis of destinations in topic_modeling.py)
- read_file.py : this file is used to read data from csv files

## Installation

before using our code, these libraries should be installed :

  - !pip install pyLDAvis
  - !pip install umap
  - !pip install stop_words
  - !pip install language_detector
  - !pip install symspellpy
  - !pip install sentence_transformers


## Usage
these are the principal parameters to change before running the code :

- the path where we store the data : path = "../data/"
- number of topics : ntopic = 5
- A list of models to apply : model = ["NMF"]
