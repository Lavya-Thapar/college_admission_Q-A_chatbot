import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, pipeline
import random

# Load the model and tokenizer
model_path = "college-queries-classification-model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict(text):
    """
    Predicts the class label for a given input text.

    Args:
        text (str): The input text for which the class label needs to be predicted.

    Returns:
        probs (torch.Tensor): Class probabilities for the input text.
        pred_label_idx (torch.Tensor): The index of the predicted class label.
        pred_label (str): The predicted class label.
    """
    # Tokenize the input text and move tensors to the GPU if available
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    # Get model output (logits)
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    
    # Get the index of the class with the highest probability
    pred_label_idx = probs.argmax()

    # Now map the predicted class index to the actual class label
    pred_label = model.config.id2label[pred_label_idx.item()]

    return probs, pred_label_idx, pred_label

#responses 

responses = {
    "admission procedure": [
        "The admission procedure involves submitting an online application, followed by an interview process.",
        "You need to complete the application form, submit your academic transcripts, and participate in a personal interview.",
        "Start by filling out the application form on our website, then submit all required documents and schedule an interview.",
        "Our admission procedure includes an online application, document verification, and an interview with the admissions committee.",
        "To begin the admission process, create an account on our portal, fill out the necessary forms, and submit your documents for review."
    ],
    "admission steps": [
        "Step 1: Complete the online application. Step 2: Submit your academic records. Step 3: Attend the interview.",
        "First, fill out the application form. Next, upload the required documents. Finally, attend an interview if shortlisted.",
        "Begin by creating a profile on our admissions portal, then follow the prompts to complete each step of the application process.",
        "Our admission steps include submitting an online application, providing academic transcripts, and participating in an interview.",
        "The steps to apply are: register on our website, fill out the application form, submit required documents, and attend an interview."
    ],
    "required documents": [
        "The required documents include your academic transcripts, letters of recommendation, a personal statement, and a copy of your ID.",
        "You'll need to provide your high school diploma, academic transcripts, letters of recommendation, and a personal statement.",
        "Required documents for admission are: academic transcripts, a resume, letters of recommendation, and a personal essay.",
        "Make sure to submit your academic records, two letters of recommendation, a personal statement, and a copy of your passport.",
        "You'll need to upload your academic transcripts, a personal statement, letters of recommendation, and a valid photo ID."
    ],
    "application fee": [
        "The application fee is $50, payable online via credit card or bank transfer.",
        "You can pay the application fee of $50 through our online portal using a credit or debit card.",
        "Our application fee is $50. Payment can be made online during the application submission process.",
        "There is a non-refundable application fee of $50 that must be paid when you submit your application.",
        "The application fee for admission is $50, which can be paid online using various payment methods."
    ],
    "application deadline": [
        "The application deadline for the upcoming semester is September 30th. Make sure to submit all documents by then.",
        "You have until September 30th to complete your application. Don't miss the deadline!",
        "Please ensure that your application is submitted by September 30th to be considered for the next intake.",
        "All applications must be received by September 30th. Late submissions will not be accepted.",
        "The last date to apply for admissions is September 30th. Ensure all materials are submitted on time."
    ]
}





# Streamlit app
st.title('College Admission Query Chatbot')

st.write("Enter a college admission related question and I will respond to it.")

# Text input
text = st.text_input("Enter your query:")

if st.button('Get answer'):
    if text:
        probs, pred_label_idx, pred_label = predict(text)
        random_response = random.choice(responses[pred_label])
        st.write(f"**Response:** {random_response}")
    else:
        st.write("Please enter a query to predict.")

# Run the app using: streamlit run app.py
