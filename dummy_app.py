from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from cleantext import clean
import contractions
from unidecode import unidecode
from bs4 import BeautifulSoup
import re


# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Load the state_dict into the model
model.load_state_dict(torch.load('distilbert50_v2.pth', map_location=device))

# Move the model to the device
model = model.to(device)

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess(text):
    # Handle unicode issues
    text = unidecode(text)

    # Clean the text
    text = clean(text,
                 to_ascii=True,                  # transliterate to closest ASCII representation
                 lower=True,                     # lowercase text
                 no_line_breaks=True,            # remove line breaks
                 no_urls=True,                   # replace URLs with a special token
                 no_emails=True,                 # replace email addresses with a special token
                 no_phone_numbers=True,          # replace phone numbers with a special token
                 no_numbers=False,
                 no_digits=False,
                 no_currency_symbols=True,       # replace currency symbols with a special token
                 no_punct=True,                  # remove punctuations
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_currency_symbol="<CUR>",
                 lang="en"
                )

    # Replace certain special characters with their string equivalents
    text = re.sub(r'%', ' percent', text)
    text = re.sub(r'$', ' dollar ', text)
    text = re.sub(r'₹', ' rupee ', text)
    text = re.sub(r'€', ' euro ', text)
    text = re.sub(r'@', ' at ', text)

    # Decontracting words
    text = contractions.fix(text)

    # Removing HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    return text

def make_prediction(question1, question2):
    # Preprocess the questions
    question1 = preprocess(question1)
    question2 = preprocess(question2)

    inputs = tokenizer.encode_plus(
        question1, question2, add_special_tokens=True, max_length=128, padding='max_length', truncation='only_second'
    )
    input_ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the predicted class index
    pred_class_idx = torch.argmax(outputs.logits).item()

    # Return a descriptive output
    if pred_class_idx == 0:
        return "The questions are not duplicates"
    else:
        return "The questions are duplicates"


# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # allow all origins to access the /predict route

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    question1 = data['question1']
    question2 = data['question2']
    prediction = make_prediction(question1, question2)
    return jsonify({'prediction': prediction})

# Endpoint for sentiment analysis (placeholder)
#@app.route('/predict_sentiment', methods=['POST'])
#def predict_sentiment():
#    data = request.get_json()
#    text = data['text']
    # prediction = make_sentiment_prediction(text)
    # return jsonify({'prediction': prediction})
#    return jsonify({'message': 'Sentiment analysis model not implemented yet'})

# Endpoint for text classification (placeholder)
#@app.route('/classify_text', methods=['POST'])
#def classify_text():
#    data = request.get_json()
#    text = data['text']
    # prediction = make_text_classification(text)
    # return jsonify({'prediction': prediction})
#    return jsonify({'message': 'Text classification model not implemented yet'})


if __name__ == '__main__':
    app.run(debug=True)
