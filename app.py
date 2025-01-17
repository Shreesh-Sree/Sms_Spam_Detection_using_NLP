import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords (cached for better performance)
@st.cache_resource
def download_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = download_stopwords()

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))  # Ensure this file exists
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Ensure this file exists

# Function to preprocess the SMS text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Join words back to string
    return " ".join(words)

# Streamlit App Configuration
st.set_page_config(page_title="SMS Spam Detection", layout="wide")

# Sidebar for Information
st.sidebar.title("Spam SMS Classifier")
st.sidebar.markdown("""
**Welcome to the SMS Spam Detection App!**  
This app classifies SMS messages as **Spam** or **Ham (Not Spam)** using machine learning.

### How it Works:
1. **Input your SMS message** in the provided text area.
2. Click on the **Submit** button to classify the SMS.
3. The result will show whether the SMS is **Spam** or **Ham**.

---

### About the Model:
- Trained on a dataset of SMS messages.
- Uses **TF-IDF Vectorization** for text representation.
- Classification is based on machine learning.

---

**Developer:** Sreesanth R
""")

# Main content area
st.title("SMS Spam Detection")
st.markdown("""
This app uses machine learning to classify SMS messages as either **Spam** or **Ham** (Not Spam).  
Enter a message below or upload a file to classify multiple messages.
""")

# Input options
sms_input = st.text_area("Enter a single SMS message:", "")
submit_button = st.button("Submit")

# File upload for bulk classification
uploaded_file = st.file_uploader("Or upload a file containing SMS messages (one per line):", type=["txt"])

# Single SMS classification
if submit_button:
    if sms_input:
        # Preprocess the input message
        cleaned_sms = preprocess_text(sms_input)
        
        # Vectorize the cleaned message
        vectorized_sms = vectorizer.transform([cleaned_sms])
        
        # Make prediction and display result
        prediction = model.predict(vectorized_sms)[0]
        probabilities = model.predict_proba(vectorized_sms)[0]

        if prediction == 1:
            st.warning(f"This message is **Spam**.")
            st.info(f"Spam Confidence: {probabilities[1]:.2%}")
        else:
            st.success(f"This message is **Not Spam (Ham)**.")
            st.info(f"Ham Confidence: {probabilities[0]:.2%}")
    else:
        st.error("Please enter an SMS message to classify.")

# Bulk SMS classification from file upload
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    messages = content.splitlines()
    
    if messages:
        st.write("Uploaded Messages:")
        st.write(messages)
        
        # Process and classify each message
        predictions = []
        for message in messages:
            cleaned_sms = preprocess_text(message)
            vectorized_sms = vectorizer.transform([cleaned_sms])
            prediction = model.predict(vectorized_sms)[0]
            predictions.append("Spam" if prediction == 1 else "Ham")
        
        # Display results
        st.write("Classification Results:")
        for msg, pred in zip(messages, predictions):
            st.write(f"**{msg}** -> {pred}")
    else:
        st.error("The uploaded file is empty. Please upload a valid file.")

# Footer
st.markdown("---")
st.markdown("""
**Note:**  
This application is for educational purposes. The accuracy of the predictions depends on the quality and size of the training dataset.
""")
