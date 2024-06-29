# STEP 1
from datasets import load_dataset, Audio
from transformers import pipeline

# STEP 2
classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model", device = 0)

# STEP 3
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]

# STEP 4
result = classifier(audio_file)

# STEP 5
print("result = ",result)