# reddit-mental-health

## Better Labeling of Reddit Mental Health Posts Using Clustering

### Research Question

**How can we better label Reddit mental health posts beyond just using subreddit names?**

- Subreddit categories (e.g., r/ADHD, r/depression) are often broad, inconsistent, or insufficient for fine-grained understanding.
- This project aims to discover **latent themes** in mental health discussions and create **more meaningful labels** for posts.

### Motivation

- **Problem**: Subreddit names alone do not capture the nuanced mental health experiences expressed in posts.
- **Goal**: Develop a data-driven method to automatically assign better, topic-based labels to Reddit mental health posts, enabling improved analysis.

### Methodology

- **Baseline Clustering**:  
  - Applied TF-IDF + KMeans clustering to estimate a reasonable number of topics.
- **Latent Dirichlet Allocation (LDA)**:
  - Trained an LDA topic model on the cleaned corpus.
  - Tuned the number of topics based on KMeans results and practical interpretability.
- **Topic Interpretation**:
  - Interpreted LDA-generated topics based on top keywords.
  - Assigned **human-readable labels** for each discovered topic.
- **Final Dataset**:
  - Created a newly labeled dataset (`labeled_data.csv`) with meaningful topic labels.

### Key Results

- Successfully identified major mental health themes:
  - Daily Life Challenges (Environment and Activities)
  - General Emotional Struggles and Reflections
  - Daily Life Challenges (Work, Sleep, Routine)
  - Personal Emotional Distress
  - ADHD and Academic Challenges
  - Timeline Narratives (Job, Life Changes)
  - PTSD and Trauma Experiences
  - Social Interaction Challenges
  - Medication Management (ADHD/Anxiety)
  - OCD and Intrusive Thoughts

- Produced a **new labeled dataset** with over 60,000 posts, improving over subreddit-based categorization.

### Contribution

- Provides a **structured and cleaner alternative labeling** for Reddit mental health posts.
- Enables **future research** such as:
  - Supervised classification models
  - Monitoring trends in mental health discussions
  - Improving NLP-based analysis of online mental health data

### Reflection

- Originally, classification modeling was considered as a next step.  
- However, the LDA clustering and human labeling process **already produced a strong deliverable** aligned with the research question: **creating a better-labeled dataset** for mental health posts.
- Future work could extend this by building supervised classifiers using the new labels or refining topic modeling with deeper models.

### Project Structure

| File/Folder | Description |
|:---|:---|
| `eda.ipynb` | Exploratory Data Analysis (EDA) of the raw Reddit posts. Preprocessing, data cleaning, initial inspection. |
| `clustering.ipynb` | Clustering and Topic Modeling pipeline: TF-IDF + KMeans baseline, LDA modeling, topic interpretation, label assignment. |
| `eda_visualizations/` | Folder containing visualization outputs from EDA (plots, charts, distributions). |
| `data_utils.py` | Utility functions for loading, cleaning, and preprocessing the Reddit dataset. |
| `lda_hyperparams_tuning.py` | (Optional script) Manual grid search code for tuning LDA hyperparameters (not used in final pipeline due to runtime). |
| `labeled_data.csv` | Final dataset with human-interpretable topic labels attached to each Reddit post. |
| `environment.yml` | Conda environment specification for dependencies (Python version, libraries). |

### Setup
```
conda env create -f environment.yml
conda activate reddit_env
```