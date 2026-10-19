import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

MODEL_PATH = Path(__file__).parent / "spam_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📩 SMS Spam Detection")
st.write("Enter a text message below. The machine-learning model will classify it as **SPAM** or **NOT SPAM**.")

message = st.text_area(
    "Message to check",
    placeholder="Example: Congratulations! You won a prize. Click here to claim...",
    height=140
)

if st.button("🔍 Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        spam_index = classes.index("spam")
        spam_score = float(probabilities[spam_index])

        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
            st.write("This message has patterns similar to spam messages in the training dataset.")
        else:
            st.success("✅ Prediction: NOT SPAM")
            st.write("This message is more similar to legitimate messages in the training dataset.")

        st.metric("Estimated spam score", f"{spam_score * 100:.1f}%")
        st.caption("This score is a model estimate, not a guarantee that a message is safe or fraudulent.")

with st.expander("Try sample messages"):
    st.code("Congratulations! You have won a free prize. Call now to claim your reward!")
    st.code("Hey, are we still meeting at the library at 4 pm?")
    st.code("URGENT! Your account has a problem. Click this link immediately!")

st.divider()
st.caption("Student project demonstration • TF-IDF + Multinomial Naive Bayes")
st.caption("Limitations: predictions can be wrong. Do not click suspicious links or share private information.")
