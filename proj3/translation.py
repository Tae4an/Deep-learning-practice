from transformers import pipeline

translator = pipeline("translation", model="google-t5/t5-base", device = 0)

text = "translate English to French: My name is Sarah and I live in London"

result = translator(text)

print("result =", result)
