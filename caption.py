# To Run: flask --app caption run
# from a command window in the C:\Users\Emmeline.Pearson\Desktop\BLIP folder
# Go to: http://127.0.0.1:5000/ to see page

from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# ------- Image pre processing ------------------------------------------------------------------------------------------
import sys
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size,device, image_file_name):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')  
    raw_image = Image.open(image_file_name).convert('RGB')   

    w,h = raw_image.size
    #(raw_image.resize((w//5,h//5))).show()
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)
    return image
# ------------------------------------------------------------------------------------------

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.post('/select')
def select():
    image_file = request.form["imageUpload"]
    
    if image_file == "" or image_file is None: 
        print("input image is invalid")
        return False

    image_size = 384
    image = load_demo_image(image_size=image_size, device=device, image_file_name=image_file)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    caption = ""

    with torch.no_grad():
        # nucleus sampling
        caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        print('caption: '+caption[0])
    

    return render_template('index.html', generated_caption=("Alt text: " + caption[0]))

