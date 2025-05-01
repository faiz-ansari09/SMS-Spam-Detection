import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Header
st.title('📧 SMS-Spam Detection')
st.write('Welcome to the SMS-Spam Detection! This application helps you determine if a SMS is spam or not.')

# Instructions
st.subheader('Instructions:')
st.write('1. Enter the SMS text in the input box below.')
st.write('2. Click the "Detect" button to check if the SMS is spam.')

# User input
user_input = st.text_area('✉️ Enter your SMS here', height=150)

# Detection button
if st.button("🔍 Detect"):
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0] == 0:
            st.success("✅ The SMS is not spam")
        else:
            st.error("🚫 The SMS is spam")
    else:
        st.warning("⚠️ Please enter a SMS to detect")

# Footer
st.write('---')
st.write('Developed by Faiz Ansari')
st.write('Contact us at ansarifaiz0905@gmail.com')
