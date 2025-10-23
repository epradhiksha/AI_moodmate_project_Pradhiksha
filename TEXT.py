from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
def detect_emotion(text):
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    predictions = emotion_classifier(text)
    pred = max(predictions[0], key=lambda x: x['score'])
    return pred['label']