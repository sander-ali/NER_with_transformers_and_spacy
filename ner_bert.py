from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

NER = pipeline("ner", model=model, tokenizer=tokenizer)
example = "I am Dr. Sunder Ali from Pakistan. My Group (AI and Machine Learning Community Pakistan) shares MATLAB and python codes along with latest news related to deep learning techniques and scholarship opportunities."

results = NER(example)
results 