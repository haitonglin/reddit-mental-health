# reddit-mental-health

## Better Labeling of Reddit Mental Health Posts Using Clustering

### Research Question

How can we better label Reddit mental health posts beyond just using subreddit names?

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

- Successfully identified major mental health themes such as:
  - ADHD and Academic Challenges
  - PTSD and Trauma Experiences
  - OCD and Intrusive Thoughts
  - Medication Management (ADHD/Anxiety)
  - Daily Life Challenges
  - General Emotional Struggles
  - Social Interaction Challenges

- Produced a **new labeled dataset** with over 60,000 posts, improving over subreddit-based categorization.

### Contribution

- Provides a **structured and cleaner alternative labeling** for Reddit mental health posts.
- Enables **future research** such as:
  - Supervised classification models
  - Monitoring trends in mental health discussions
  - Improving NLP-based analysis of online mental health data


## Reflection

- Originally, classification modeling was considered as a next step.  
- However, the LDA clustering and human labeling process **already produced a strong deliverable** aligned with the research question: **creating a better-labeled dataset** for mental health posts.
- Future work could extend this by building supervised classifiers using the new labels or refining topic modeling with deeper models.