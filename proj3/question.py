from transformers import pipeline

question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2", device = 0)

question = 'Why is model conversion important?'
context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'

result =question_answerer(question=question, context=context)

print("result = ", result)