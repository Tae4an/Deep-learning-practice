# STEP 1
from transformers import pipeline

# STEP 2
classifier = pipeline('text-classification', model="vectara/hallucination_evaluation_model", device = 0)

# STEP 3
text = "A man walks into a bar and buys a drink [SEP] A bloke swigs alcohol at a pub"

# STEP 4
result = classifier(text)

# STEP 5
print(result)