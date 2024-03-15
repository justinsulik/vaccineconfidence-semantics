
# Overview: vaccine semantics

This repository contains initial/draft code for one component of a project analysing online representations of vaccines and complementary medicines. It was written by https://github.com/fabianbeigang (https://github.com/fabianbeigang/vaccine-attitude-semantics), but is cloned here for easier connection with the other project components (ultimately at https://osf.io/wmnuk/, though currently private until appropriate data privacy and integrity checks have been carried out). 

This component focuses on the semantics of terms related to "Vaccine" across diverse online sources (e.g., mainstream vs pseudoscience). 

The corpus analysed here consists of:

- Texts from websites known to endorse complementary and alternative medicines (CAM)
- Texts from trusted journalistic sources (TRUSTED)
- Texts from random search engine results (SE)

Note that the corpus itself cannot be shared openly with the public, but the scripts for finding and scraping the URLs are available at https://github.com/justinsulik/vaccineconfidence-webscraping and https://github.com/burcegumuslu/vh_corpuscode. 

# Contents

The repository contains various notebooks (numbered as follows):

- 1.0. Data pre-prcessing
- 1.1. A descriptive overview of the data
- 2.0. Word2vec embedding model of the corpus
- 2.1. Exploration of the embeddings, with a specific focus on semantic differences wrt vaccine-related terms across the corpora
- 3.0. a BERTopic model

In addition utility functions are in utils.py



