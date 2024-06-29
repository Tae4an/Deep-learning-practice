from transformers import pipeline

classifier = pipeline("ner", model="Clinical-AI-Apollo/Medical-NER",device = 0)

text = "45 year old woman diagnosed with CAD"

result = classifier(text)

# STEP 5
print("결과 = ",result)