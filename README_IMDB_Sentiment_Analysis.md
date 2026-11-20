# IMDb Movie Review Sentiment Analysis using NLP and Machine Learning

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies IMDb movie reviews as **Positive** or **Negative**. The project includes text cleaning, TF-IDF feature extraction, model training and comparison, performance evaluation, error inspection, and sentiment prediction for custom reviews.

This project was developed as **CodeAlpha Data Analytics Internship – Task 4: Sentiment Analysis**.

<p align="center">
  <a href="https://github.com/Harischandra-Prasad-Vissamsetti/IMDb-Movie-Review-Sentiment-Analysis">
    <strong>View GitHub Repository</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">
    <strong>Download Dataset from Kaggle</strong>
  </a>
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Dataset Information](#dataset-information)
- [Repository Structure](#repository-structure)
- [Access the Project from GitHub](#access-the-project-from-github)
- [Download the Dataset from Kaggle](#download-the-dataset-from-kaggle)
- [Run the Project in Google Colab](#run-the-project-in-google-colab)
- [Run the Project Locally](#run-the-project-locally)
- [Project Workflow](#project-workflow)
- [Text Preprocessing](#text-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Custom Sentiment Prediction](#custom-sentiment-prediction)
- [Results and Interpretation](#results-and-interpretation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

## Project Overview

Online movie reviews contain useful opinions about storylines, acting, direction, and overall viewing experiences. Reading and classifying a large number of reviews manually can be time-consuming.

This project applies NLP and supervised machine learning to classify review text into two sentiment categories:

- **Positive** – the review expresses a favorable opinion.
- **Negative** – the review expresses an unfavorable opinion.

The notebook uses a labeled IMDb movie review dataset from Kaggle. It prepares the text, converts it into numerical features using TF-IDF, trains two classification models, compares their results, and uses the selected model to predict sentiment for new review text.

### Project Details

| Item | Description |
|---|---|
| Project name | IMDb Movie Review Sentiment Analysis |
| Internship | CodeAlpha – Data Analytics |
| Task | Task 4: Sentiment Analysis |
| Domain | Natural Language Processing (NLP) |
| Project type | Binary Text Classification |
| Language | Python |
| Environment | Google Colab / Jupyter Notebook |
| Dataset source | Kaggle – IMDb Dataset of 50K Movie Reviews |

## Objectives

- Load and inspect a labeled movie review dataset.
- Validate the required columns and sentiment labels.
- Identify and remove blank and duplicate review records.
- Clean raw review text for NLP processing.
- Convert text into numerical features using TF-IDF.
- Train Logistic Regression and Multinomial Naive Bayes classifiers.
- Compare models using classification metrics.
- Evaluate the selected model with a classification report and confusion matrix.
- Inspect incorrectly classified reviews.
- Predict the sentiment of user-provided movie reviews.

## Key Features

- Dataset validation and preparation.
- Positive and negative sentiment distribution visualization.
- HTML tag and URL removal from review text.
- Text normalization and whitespace cleanup.
- Stratified 80/20 train-test split.
- TF-IDF vectorization with unigram and bigram features.
- Logistic Regression and Multinomial Naive Bayes model pipelines.
- Model comparison using accuracy, precision, recall, F1-score, and macro-F1.
- Classification report and confusion matrix.
- Misclassified review inspection.
- Custom review sentiment prediction with class probabilities.

## Technology Stack

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Text vectorization, model training, splitting, and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical charts and confusion matrix visualization |
| Joblib | Imported for model-related utility support |
| Google Colab | Cloud notebook execution |
| Jupyter Notebook | Interactive development environment |

## Dataset Information

**Dataset:** [IMDb Dataset of 50K Movie Reviews – Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The project uses the Kaggle IMDb movie review dataset. The notebook expects a CSV file named:

```text
IMDB Dataset.csv
```

The required dataset columns are:

| Column | Description |
|---|---|
| `review` | Text content of the movie review |
| `sentiment` | Sentiment label: `positive` or `negative` |

The notebook filters the sentiment column to the two supported labels, removes blank reviews and duplicate review-label records, and proceeds with the usable records.

### Why the dataset is not included in this repository

The dataset CSV is intentionally not stored in this GitHub repository because of its large file size. Keeping the notebook and source documentation in GitHub while linking to the original Kaggle dataset makes the repository easier to clone and maintain.

**You must download the CSV from Kaggle separately before running the notebook.** A Kaggle account may be required to download the dataset. Please review the dataset page's license and terms before redistributing or using the data commercially.

## Repository Structure

```text
IMDb-Movie-Review-Sentiment-Analysis/
│
├── README.md
│
└── IMDb_Sentiment_Analysis_Colab_File.ipynb
```

The dataset is downloaded separately and is not part of the GitHub repository.

## Access the Project from GitHub

### Option 1: Download the notebook from GitHub

1. Open the project repository:

   [IMDb Movie Review Sentiment Analysis – GitHub](https://github.com/Harischandra-Prasad-Vissamsetti/IMDb-Movie-Review-Sentiment-Analysis)

2. Find and open `IMDb_Sentiment_Analysis_Colab_File.ipynb`.
3. Click the **Download raw file** option to download the notebook to your computer.
4. Open [Google Colab](https://colab.research.google.com/).
5. Select **File → Upload notebook** and choose the downloaded `.ipynb` file.

### Option 2: Clone the repository using Git

Make sure Git is installed, then open a terminal or command prompt and run:

```bash
git clone https://github.com/Harischandra-Prasad-Vissamsetti/IMDb-Movie-Review-Sentiment-Analysis.git
```

Open the downloaded project folder:

```bash
cd IMDb-Movie-Review-Sentiment-Analysis
```

The notebook can then be opened using Google Colab, Jupyter Notebook, or a compatible notebook editor.

## Download the Dataset from Kaggle

The dataset is not stored in this repository. Follow these steps to obtain it:

1. Open the direct Kaggle dataset page:

   **[IMDb Dataset of 50K Movie Reviews – Download on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**

2. Sign in to your Kaggle account, or create a free account if you do not have one.
3. Click the **Download** option on the dataset page.
4. If Kaggle downloads a ZIP archive, extract it.
5. Locate the CSV file named `IMDB Dataset.csv`.
6. Keep the filename unchanged, or update the notebook's data-loading path if you rename it.

After downloading, you should have:

```text
IMDB Dataset.csv
```

## Run the Project in Google Colab

Google Colab is the recommended option for running this project because the notebook uses Google Colab's file system path.

### Step 1: Open the notebook

1. Visit [Google Colab](https://colab.research.google.com/).
2. Sign in with your Google account.
3. Select **File → Upload notebook**.
4. Upload `IMDb_Sentiment_Analysis_Colab_File.ipynb` downloaded from GitHub.

### Step 2: Upload the Kaggle dataset

1. Download and extract `IMDB Dataset.csv` from Kaggle as explained above.
2. In the Colab notebook, open the **Files** panel on the left.
3. Click **Upload to session storage**.
4. Select `IMDB Dataset.csv` from your computer.
5. Confirm that the file appears in the Colab `/content/` directory.

### Step 3: Check the dataset path

The notebook's active data-loading cell uses:

```python
raw_df = pd.read_csv("/content/IMDB Dataset.csv", encoding="latin-1")
```

Therefore, the uploaded CSV must be available at exactly:

```text
/content/IMDB Dataset.csv
```

If the file has a different name or is stored in another folder, update the path in the notebook's `pd.read_csv()` statement.

**Note:** The notebook contains an interactive `files.upload()` code example in comments. The active loading statement uses the `/content/IMDB Dataset.csv` path. The simplest approach is to upload the file through the Colab Files panel and keep the expected filename.

### Step 4: Install the required libraries

Run the notebook's setup cell. It installs the required packages:

```python
!pip -q install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 5: Run all notebook cells

1. Select **Runtime → Run all**, or execute the cells one by one from top to bottom.
2. Wait for data loading, preprocessing, model training, and evaluation to finish.
3. Review the sentiment distribution chart, model comparison table, classification report, and confusion matrix.
4. Try the custom review prediction cell with your own movie review.

**Important:** Colab session storage is temporary. If the runtime disconnects or resets, upload the CSV again before rerunning the notebook.

## Run the Project Locally

You can also execute the notebook on your computer with Python and Jupyter installed.

### Step 1: Clone the repository

```bash
git clone https://github.com/Harischandra-Prasad-Vissamsetti/IMDb-Movie-Review-Sentiment-Analysis.git
cd IMDb-Movie-Review-Sentiment-Analysis
```

### Step 2: Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

### Step 4: Download the dataset

Download the CSV from the [Kaggle dataset page](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project folder.

The notebook's current data-loading path is `/content/IMDB Dataset.csv`, which is specific to Google Colab. When running locally, change that line to:

```python
raw_df = pd.read_csv("IMDB Dataset.csv", encoding="latin-1")
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `IMDb_Sentiment_Analysis_Colab_File.ipynb` in the Jupyter interface and execute the cells in order.

## Project Workflow

The notebook follows these stages:

1. **Environment setup:** Install and import the required Python libraries.
2. **Dataset loading:** Read the Kaggle CSV file into a Pandas DataFrame.
3. **Data validation:** Check that `review` and `sentiment` columns are present.
4. **Data preparation:** Normalize sentiment labels, keep positive and negative reviews, and remove blank and duplicate records.
5. **Sentiment distribution:** Visualize the number of positive and negative reviews.
6. **Text cleaning:** Remove HTML markup and URLs, retain letters and apostrophes, normalize whitespace, and convert text to lowercase.
7. **Train-test split:** Split the data into training and testing sets using an 80/20 stratified split.
8. **Feature extraction:** Convert review text into numerical features using TF-IDF.
9. **Model training:** Train Logistic Regression and Multinomial Naive Bayes pipelines.
10. **Model comparison:** Compare both models using holdout test-set metrics.
11. **Model evaluation:** Generate a classification report and confusion matrix for the selected model.
12. **Error analysis:** Inspect examples where the selected model predicts the wrong sentiment.
13. **Custom prediction:** Predict the sentiment of a new movie review and display class probabilities.

## Text Preprocessing

The `clean_text()` function prepares raw review text using the following operations:

- Converts HTML entities into their corresponding characters.
- Removes HTML tags.
- Removes URLs.
- Keeps alphabetic characters and apostrophes.
- Replaces repeated whitespace with a single space.
- Removes leading and trailing spaces.
- Converts text to lowercase.

The notebook does **not** remove all stopwords. Words such as **“not”** can change the meaning and sentiment of a review, so retaining them is important.

## Machine Learning Models

The project trains and compares two supervised classification algorithms.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm used to estimate the probability that a review belongs to a sentiment class.

The notebook uses a TF-IDF vectorizer and a Logistic Regression classifier with:

- Maximum iterations: `1500`
- Class weighting: `balanced`
- TF-IDF maximum features: `30000`
- N-gram range: `(1, 2)`

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks.

The notebook uses a TF-IDF vectorizer and a Multinomial Naive Bayes classifier with:

- Smoothing parameter (`alpha`): `0.5`
- TF-IDF maximum features: `30000`
- N-gram range: `(1, 2)`

### Why use a Scikit-learn Pipeline?

Each model combines TF-IDF vectorization and classification in a single Scikit-learn pipeline. This ensures that the TF-IDF vocabulary and inverse document frequency values are fitted on the training data only, rather than learning from the held-out test set.

## Model Evaluation

The dataset is divided using:

| Dataset portion | Percentage |
|---|---:|
| Training set | 80% |
| Testing set | 20% |

The split is stratified to preserve the sentiment class proportions, and `random_state=42` is used for reproducibility.

The notebook evaluates both models using:

| Metric | Meaning |
|---|---|
| Accuracy | Overall proportion of correctly classified reviews |
| Precision (Positive) | Proportion of predicted-positive reviews that are actually positive |
| Recall (Positive) | Proportion of actual-positive reviews correctly identified |
| F1-score (Positive) | Harmonic mean of positive-class precision and recall |
| Macro-F1 | Average F1-score across positive and negative classes, giving each class equal weight |

The model with the highest **holdout macro-F1** is selected for the detailed classification report, confusion matrix, error inspection, and custom predictions.

### Evaluation visualizations

- **Classification report:** Displays precision, recall, F1-score, and support for each sentiment class.
- **Confusion matrix:** Shows correct and incorrect predictions for positive and negative reviews.
- **Model comparison table:** Compares the evaluation metrics for Logistic Regression and Multinomial Naive Bayes.

No fixed accuracy or performance value is claimed in this README. Run the notebook to generate the metrics for your dataset and environment.

## Custom Sentiment Prediction

After the models have been trained, the notebook defines a `predict_sentiment()` function that accepts a review as input and returns the predicted sentiment and class probabilities.

Example:

```python
sample_review = "The story was engaging and the performances were excellent."

prediction, probabilities = predict_sentiment(sample_review)

print("Predicted sentiment:", prediction.upper())
print("Class probabilities:", probabilities)
```

You can also replace the sample text in the notebook's **Enter a custom review** cell with your own movie review.

## Results and Interpretation

Use the output generated by your notebook execution to discuss:

- Which model achieved the higher macro-F1 score.
- How well each model identifies positive and negative reviews.
- What the confusion matrix reveals about false positive and false negative predictions.
- What patterns appear in the misclassified examples.
- Whether the selected model is suitable for the intended review-classification task.

Model performance can vary depending on the dataset version, environment, and changes to preprocessing or model settings. Always report results from an actual notebook run.

## Limitations

- The model is trained on IMDb movie reviews; performance may differ on product reviews, social media posts, slang, or other languages.
- The project supports only positive and negative labels; it does not classify neutral or mixed sentiment.
- TF-IDF models do not fully understand sarcasm, context, or subtle language.
- The class probabilities returned by the model should not automatically be interpreted as calibrated confidence scores.
- The notebook's model and output saving statements are commented out; the current workflow does not automatically export trained models or evaluation files.

## Future Enhancements

- Add a neutral sentiment class.
- Compare additional models such as Linear SVM.
- Experiment with transformer-based models such as DistilBERT.
- Add cross-validation on the training data while preserving a final untouched test set.
- Build an interactive sentiment prediction interface using Streamlit.
- Add batch prediction for multiple reviews from a CSV file.
- Evaluate model performance on separately labeled review datasets from other domains.
- Add reusable model and evaluation export functionality.

## Author

**Harischandra Prasad Vissamsetti**

- GitHub Profile: [Harischandra-Prasad-Vissamsetti](https://github.com/Harischandra-Prasad-Vissamsetti)
- Project Repository: [IMDb-Movie-Review-Sentiment-Analysis](https://github.com/Harischandra-Prasad-Vissamsetti/IMDb-Movie-Review-Sentiment-Analysis)

## Acknowledgements

- **CodeAlpha** – Data Analytics Internship project task.
- **Kaggle** – For hosting the IMDb Dataset of 50K Movie Reviews.
- **Scikit-learn and Python open-source communities** – For the tools and libraries used in this project.

**Dataset attribution:** Lakshmipathi N, *IMDb Dataset of 50K Movie Reviews*, Kaggle. Refer to the [original Kaggle dataset page](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) for dataset details and applicable terms.

---

If you find this project useful, consider giving the repository a star. Suggestions and constructive feedback are welcome.
