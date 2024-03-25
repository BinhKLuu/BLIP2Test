import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import time
#import torch

start = time.time()
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
#model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
end_load = time.time()
print("Done loading model.")
print(end_load-start)

#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)

url = r"https://m.media-amazon.com/images/I/61fXJ6AzFKL._AC_SX466_.jpg"
get_img = requests.get(url, stream=True)
image = Image.open(get_img.raw).convert('RGB')
image.resize((596, 437)).show()

inputs = processor(image, return_tensors="pt")
#inputs = processor(image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
end_process = time.time()
print(generated_text)
print(end_process-end_load)

# Q&A Session
while(1):
    question = input("\nMe: ")
    if question == "new":
        
        while (question == "new"):
            url = input("\nNew Image: ")
            try:
                get_img = requests.get(url, stream=True)
                image = Image.open(get_img.raw).convert('RGB')
                image.resize((596, 437)).show()
                question = ""
            except Exception as err:
                print("Error loading image.")
        prompt = ""
    else:
        prompt = "Question: " + question + " Answer: "

    start_q = time.time()
    inputs = processor(image, text=prompt, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("BLIP2: " + generated_text)
    end_q = time.time()
    print(end_q-start_q)