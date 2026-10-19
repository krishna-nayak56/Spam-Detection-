# SMS Spam Detection — Machine Learning Mini Project

A simple, runnable college demonstration that classifies a text message as **SPAM** or **NOT SPAM**.

## Machine-learning workflow
1. Load labeled SMS messages.
2. Split messages into training and test sets.
3. Convert text into numerical TF-IDF features.
4. Train a Multinomial Naive Bayes classifier.
5. Evaluate it on held-out test data.
6. Use the trained pipeline in a Streamlit interface.

## Files
- `app.py` — web interface
- `train_model.py` — training and evaluation script
- `requirements.txt` — Python dependencies
- `spam_model.joblib` — trained model (created after running training)
- `SMSSpamCollection` — optional local dataset file; training script downloads the public dataset if absent

## Run on a laptop (Python 3.10+ recommended)
Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

Train and evaluate the model:
```bash
python train_model.py
```

Launch the app:
```bash
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`, to open in your browser.

## If internet access is unavailable
The training script falls back to a tiny built-in sample dataset so the app can still run. This fallback is for demonstration only and has very limited generalization. For a better model, download the UCI SMS Spam Collection dataset and place the extracted `SMSSpamCollection` file in this folder before training.

## Example messages
Spam:
> Congratulations! You have won a free prize. Call now to claim.

Not spam:
> Hey, are we still meeting at the library at 4 pm?

## Viva explanation
- **What is spam detection?** A text-classification task that labels messages as spam or legitimate.
- **What is TF-IDF?** A method that gives words numerical weights based on how informative they are in a document relative to a collection of documents.
- **Why Naive Bayes?** It is fast, simple, and often works well for text classification.
- **Why split the dataset?** To evaluate performance on messages the model did not train on.
- **What is accuracy?** The fraction of test examples classified correctly.
- **What is precision?** Among messages predicted as spam, the fraction that really are spam in the labeled test data.
- **What is recall?** Among actual spam messages in the test data, the fraction detected by the model.
- **Limitations?** It can misclassify messages, struggle with new scams or different languages, and its score is not proof that a message is safe or fraudulent.

## Important note
This is an educational prototype, not a production security filter. Never click a suspicious link just because this app labels the message as not spam.
