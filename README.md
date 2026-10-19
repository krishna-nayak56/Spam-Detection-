# 📩 SMS Spam Detection Using Machine Learning

A Machine Learning project that classifies SMS messages as **SPAM** or **NOT SPAM** using Natural Language Processing (NLP).

This project uses **TF-IDF** for text feature extraction and **Multinomial Naive Bayes** for classification. A simple interactive web application built with Streamlit allows users to enter messages and check whether they are potentially spam.

## 🚀 Features

- 📩 Classifies SMS messages as SPAM or NOT SPAM.
- 🤖 Uses Machine Learning for text classification.
- 🔤 Uses TF-IDF to convert text into numerical features.
- 📊 Evaluates model performance using standard metrics.
- 🌐 Provides an interactive web interface using Streamlit.
- 📈 Displays an estimated spam probability score.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data handling
- **Scikit-learn** — Machine Learning
- **TF-IDF Vectorizer** — Text feature extraction
- **Multinomial Naive Bayes** — Classification algorithm
- **Streamlit** — Web application
- **Joblib** — Saving and loading the trained model

## 📁 Project Structure

```text
Spam-Detection-/
│
├── sms_spam_detection_project/
│   ├── app.py
│   ├── train_model.py
│   ├── SMSSpamCollection
│   ├── spam_model.joblib
│   ├── sample_messages.csv
│   ├── requirements.txt
│   └── README.md
│
└── .gitignore
```

**Main files:**

| File | Purpose |
|---|---|
| `app.py` | Runs the Streamlit web application |
| `train_model.py` | Trains and evaluates the ML model |
| `SMSSpamCollection` | Dataset containing labelled SMS messages |
| `spam_model.joblib` | Stores the trained model pipeline |
| `sample_messages.csv` | Contains example messages for reference |
| `requirements.txt` | Lists the required Python libraries |
| `README.md` | Project documentation |

## ⚙️ How It Works

The project follows these steps:

1. **Data Collection:** Load the SMS dataset containing spam and legitimate messages.
2. **Data Preparation:** Separate messages and their labels.
3. **Feature Extraction:** Apply TF-IDF to convert text into numerical features.
4. **Model Training:** Train a Multinomial Naive Bayes classifier.
5. **Model Evaluation:** Evaluate the model using a test dataset.
6. **Model Saving:** Save the trained pipeline using Joblib.
7. **Prediction:** The Streamlit application loads the saved model and predicts whether a new message is spam.

### 🔄 Workflow

```text
SMS Dataset
     ↓
Data Preparation
     ↓
TF-IDF Feature Extraction
     ↓
Multinomial Naive Bayes
     ↓
Model Evaluation
     ↓
Save Trained Model
     ↓
Streamlit Web Application
     ↓
SPAM / NOT SPAM Prediction
```

## 📊 Model Performance

In one local test run, the model achieved the following results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 97.1% |
| Spam Precision | 100% |
| Spam Recall | 77% |
| Spam F1-Score | 87% |

**Note:** These results are from a particular training run and may vary depending on the dataset and train-test split.

- **Accuracy:** Percentage of all predictions that were correct.
- **Precision:** How many messages predicted as spam were actually spam.
- **Recall:** How many actual spam messages the model successfully detected.
- **F1-Score:** A combined measure of precision and recall.

Although the model achieved high accuracy, it may still miss some spam messages.

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/krishna-nayak56/Spam-Detection-.git
```

Navigate into the project directory:

```bash
cd Spam-Detection-/sms_spam_detection_project
```

### 2. Install Dependencies

Make sure Python is installed on your system.

Run:

```bash
python -m pip install -r requirements.txt
```

### 3. Train the Model (Optional)

If you want to retrain the model using the included dataset, run:

```bash
python train_model.py
```

The trained model will be saved as `spam_model.joblib`.

### 4. Run the Web Application

Run the following command from the directory containing `app.py`:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

You can now enter SMS messages and test the model.

## 🧪 Example Predictions

### Example 1: Spam Message

**Input:**

```text
Congratulations! You have won a free prize. Click here to claim now!
```

**Expected classification:** `SPAM`

### Example 2: Legitimate Message

**Input:**

```text
Hey, are we still meeting at 6 pm?
```

**Expected classification:** `NOT SPAM`

*Actual predictions depend on the trained model and the exact message.*

## ⚠️ Limitations

- The model may incorrectly classify some messages.
- It may miss spam messages that do not resemble the training data.
- Its performance may vary on modern spam messages and messages in other languages.
- The estimated spam score is not a guarantee that a message is malicious.
- The model is intended for educational purposes and should not be treated as a complete security solution.

## 🔮 Future Improvements

- Train the model using a larger and more recent dataset.
- Compare Naive Bayes with Logistic Regression and Support Vector Machines.
- Add a confusion matrix and classification report.
- Improve text preprocessing.
- Support multiple languages.
- Deploy the application online.

## 🎓 Learning Outcomes

Through this project, I learned how to:

- Prepare labelled text data.
- Apply NLP techniques for text classification.
- Convert text into numerical features using TF-IDF.
- Train and evaluate a Machine Learning model.
- Save and load a trained model using Joblib.
- Build an interactive web application using Streamlit.

## 👨‍💻 Author

**Krishna Nayak**

Machine Learning project developed for educational purposes.

---

⭐ If you find this project useful, consider giving the repository a star!
