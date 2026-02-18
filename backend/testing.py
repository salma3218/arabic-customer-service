from transformers import AutoTokenizer, AutoModelForMaskedLM
from arabert.preprocess import ArabertPreprocessor
import torch

# 1. Define the model name
model_name = "aubmindlab/bert-base-arabertv2"

# 2. Initialize the specific Preprocessor for AraBERTv2
# CRITICAL STEP: AraBERTv2 requires Farasa segmentation. 
# The ArabertPreprocessor handles this automatically.
arabert_prep = ArabertPreprocessor(model_name=model_name)

# 3. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 4. Define a sentence with a Mask
# Example: "The capital of France is [MASK]" -> "عاصمة فرنسا هي [MASK]"
text = "عاصمة فرنسا هي [MASK]"

# 5. Preprocess the text
# This applies Farasa segmentation (e.g., might split words like 'frnsa' -> 'frns +a')
text_preprocessed = arabert_prep.preprocess(text)
print(f"Preprocessed Text: {text_preprocessed}")

# 6. Tokenize
inputs = tokenizer(text_preprocessed, return_tensors="pt")

# 7. Predict
with torch.no_grad():
    outputs = model(**inputs)
    
# 8. Decode the prediction
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
predicted_word = tokenizer.decode(predicted_token_id)

print(f"Original Text: {text}")
print(f"Predicted Word: {predicted_word}")