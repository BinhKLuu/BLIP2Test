import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
#import torch

start = time.time()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
end = time.time()
print("Done loading model " + str(end-start))

img_url = r'https://media.istockphoto.com/id/1292475721/photo/house-construction-framing-gradating-into-finished-kitchen-build.jpg?s=2048x2048&w=is&k=20&c=pNyU0cGL5pZK_svbvpG4hYtbWLktbO-gJo86buAhLvA='
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
raw_image.show()

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
print("UNCONDITIONAL: " + processor.decode(out[0], skip_special_tokens=True))

# conditional image captioning
#text = "status of completion of the house"
#inputs = processor(raw_image2, text, return_tensors="pt")
#out = model.generate(**inputs)
#print("CONDITIONAL: " + processor.decode(out[0], skip_special_tokens=True))

while(1):
    prompt = input("\nMe: ")
    start_q = time.time()
    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("BLIP2: " + generated_text)
    end_q = time.time()
    print(end_q-start_q)