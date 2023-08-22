# TrOCR model for OCR on printed text

# Load the model and its tokenizer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

model_version = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(model_version)
model = VisionEncoderDecoderModel.from_pretrained(model_version)


img_path1 = "./data/act_time_country.png"

image = Image.open(img_path1).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
extract_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("output: ", extract_text)


crp_image = image.crop((750, 3.4, 970, 33.94))
display(crp_image)
display(image)

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
extract_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(extract_text)
