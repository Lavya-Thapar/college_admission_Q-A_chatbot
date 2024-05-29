import pandas as pd
import torch
import json
import random
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import streamlit as st

data = {
    "patterns": [
        "How do I apply to college?", "What are the admission requirements?", "When is the application deadline?",
        "What documents are needed for admission?", "How much is the application fee?",
        "Hi", "Hello", "Hey", "Goodbye", "Bye", "See you later", "Thank you", "Thanks"
    ],
    "tags": [
        "admission_procedure", "admission_requirements", "application_deadline",
        "required_documents", "application_fee",
        "greeting", "greeting", "greeting", "goodbye", "goodbye", "goodbye", "thanks", "thanks"
    ]
}

df = pd.DataFrame(data)



# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['tags'].unique()))

# Encode the labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['tags'])

# Create the dataset
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create dataset and dataloader
dataset = IntentDataset(
    texts=df['patterns'].to_numpy(),
    labels=df['label'].to_numpy(),
    tokenizer=tokenizer,
    max_len=32
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

with open('college_details.json', 'r') as file:
    college_data = json.load(file)

# Helper function to classify intent
def classify_intent(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=32,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(outputs.logits, dim=1)

    return le.inverse_transform(prediction.detach().numpy())[0]

# Define the chatbot responses
responses = {
    "greeting": ["Hi there!", "Hello!", "Hey! How can I assist you today?"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "thanks": ["You're welcome!", "No problem!", "Glad I could help!"]
}

# Streamlit chatbot interface
# Streamlit chatbot interface
def main():
    st.title("College Admission Chatbot")
    st.write("Ask me anything about college admissions! I can provide information about application procedures, requirements, deadlines, and more.")

    # Initialize session state variables if not already set
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_college" not in st.session_state:
        st.session_state.current_college = None
    if "waiting_for_college" not in st.session_state:
        st.session_state.waiting_for_college = False

    user_input = st.text_input("You:", "")

    if user_input:
        mentioned_college = None
        for college in college_data:
            if college.lower() in user_input.lower():
                mentioned_college = college
                break

        if mentioned_college:
            st.session_state.current_college = mentioned_college
            response = f"Please ask your specific question regarding {mentioned_college}."
            st.session_state.history.append({"user": user_input, "bot": response})
        else:
            intent = classify_intent(user_input)
            if intent == "greeting":
                response = random.choice(responses[intent])
            elif intent == "goodbye":
                response = random.choice(responses[intent])
                st.session_state.history.append({"user": user_input, "bot": response})
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()
            elif intent == "thanks":
                response = random.choice(responses[intent])
            elif intent in ["admission_procedure", "admission_requirements", "application_deadline", "required_documents", "application_fee"]:
                if st.session_state.current_college:
                    specific_info = college_data[st.session_state.current_college].get(intent, "I'm sorry, I don't have that information.")
                    response = specific_info
                else:
                    st.session_state.waiting_for_college = True
                    response = "Please specify the college from the following: " + ", ".join(college_data.keys())
            else:
                response = "I'm sorry, I don't understand that question. Can you please rephrase?"

            st.session_state.history.append({"user": user_input, "bot": response})
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key="chatbot_response")

    if st.button("Show Conversation History"):
        st.write("Conversation History:")
        for entry in st.session_state.history:
            st.write(f"You: {entry['user']}")
            st.write(f"Chatbot: {entry['bot']}")

    st.write("Conversation History:")
    for entry in st.session_state.history:
        st.write(f"You: {entry['user']}")
        st.write(f"Chatbot: {entry['bot']}")

if __name__ == '__main__':
    main()
