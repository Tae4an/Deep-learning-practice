# STEP 1
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

# STEP 2
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
model.ot("cuda:0")

# STEP 3
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# STEP 4
# 4-1 pre=processing : data to tensor
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
# 4-2 inference
with torch.no_grad():
    logits = model(**inputs).logits
# 4-3 post-processing
predicted_class_id = logits.argmax().item()
result = model.config.id2label[predicted_class_id]

# STEP 5
print(result)
