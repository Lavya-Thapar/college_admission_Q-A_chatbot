**Model**
This chatbot uses a pretrained BERT model 

**Preprocessing**
The BertTokenizer is used to tokenize the textual information and the intents of responses are Label Encoded

**Dataset**
A customized dataset for user inputs and their corresponding responses has been prepared for training purpose

**Knowledge Base**
A college_details.json file is loaded holding necessary information according to each intent for different colleges

**Interface**
A streamlit interface is designed to integrate with the model at backend, has an input box and the chatbot response box. The conversation history can also be seen in the interface as well.
