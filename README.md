# Better Labeling of Reddit Mental Health Posts: Analysis with Clustering, Topic Modeling and Human Interpretation

## Setup
```
conda env create -f environment.yml -n reddit_env
conda activate reddit_env
```

## Project Structure

| File/Folder | Description |
|:---|:---|
| `eda.ipynb` | Exploratory Data Analysis (EDA) of the raw Reddit posts. Preprocessing, data cleaning, initial inspection |
| `main.ipynb` | Clustering and Topic Modeling pipeline: TF-IDF + KMeans baseline, LDA modeling, topic interpretation, label assignment |
| `data_utils.py` | Utility functions for loading, cleaning, and preprocessing the Reddit dataset |
| `lda_hyperparams_tuning.py` | (Optional script) Manual grid search code for tuning LDA hyperparameters (not used in final pipeline due to runtime) |
| `processed_data.csv` | Dataset after preprocessing (removed in this repo because file was too big but will be here if you re-run) |
| `labeled_data.csv` | Final dataset with human-interpretable topic labels attached to each Reddit post |
| `eda_visualizations/` | Folder containing visualization outputs from EDA (plots, charts, distributions) |
| `results_visualizations/` | Folder containing visualizations from the last part of main.ipynb |
| `lda_vis.html` | Web file containing LDA topic modeling visualization |
| `environment.yml` | Conda environment specification for dependencies (Python version, libraries) |

## Research Question

**How can we better label Reddit mental health posts beyond just using subreddit names?**

- Subreddit categories (e.g., r/ADHD, r/depression) are often broad, inconsistent, or insufficient for fine-grained understanding, the names alone do not capture the nuanced mental health experiences expressed in posts.
- This project aims to discover **latent themes** in mental health discussions and create **more meaningful labels** for posts. In other words, the goal is to develop a data-driven method to automatically assign better, topic-based labels to Reddit mental health posts, enabling improved analysis.

## Methodology

- **EDA**:  
  - Preprocessing, data cleaning, initial inspection on raw data
- **Baseline Clustering**:  
  - Applied TF-IDF + KMeans clustering to estimate a reasonable number of topics
- **Latent Dirichlet Allocation (LDA)**:
  - Trained an LDA topic model on the cleaned corpus
  - Select the number of topics based on KMeans results and practical interpretability
- **Topic Interpretation**:
  - Interpreted LDA-generated topics based on top keywords
  - Assigned **human-readable labels** for each discovered topic
- **Final Dataset**:
  - Created a newly labeled dataset (`labeled_data.csv`) with meaningful topic labels

## Key Results

- Successfully identified major mental health themes/labels:
  - 0: "Physical/Sexual Trauma",
  - 1: "Sleep Disruption & Panic Symptons",
  - 2: "Work Avoidance and Exhaustion",
  - 3: "Identity and Emotional Denial",
  - 4: "Flashbacks & Avoidance Behaviors",
  - 5: "General Emotional Struggle",
  - 6: "OCD & Intrusive Thoughts",
  - 7: "PTSD Diagnosis & Mental Health Services",
  - 8: "Medication & Psychiatric Care",
  - 9: "Family & Developmental History"

- Produced a **new labeled dataset** with over 60,000 posts, improving over subreddit-based categorization
- Enables future research such as:
  - Supervised classification models
  - Monitoring trends in mental health discussions
  - Improving NLP-based analysis of online mental health data

## Discussion

- Originally, classification/self-supervised learning was considered as a next step
- However, the LDA clustering and human labeling process are suffient for the research question
- And it would not make sense to predict the labels I just created