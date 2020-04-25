import torch
from PIL import Image
import numpy as np
import argparse
import json
from my_module import Network

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("checkpoint")
parser.add_argument("--top_k", dest="topk", type=int, default=1)
parser.add_argument("--category_names", dest="categories")
parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)

args = parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image.thumbnail((256, 256))
    left = (image.height - 224) / 2
    upper = (image.width - 224) / 2
    image = image.crop((left, upper, left+224, upper+224))
    
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    return torch.from_numpy(np_image.transpose((2, 0, 1)))

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = Image.open(image_path)
    image = process_image(image)
    images = torch.unsqueeze(image, 0).type(torch.FloatTensor)
    
    top_p, top_class = model.predict(images, topk)
    
    idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    probs = top_p.cpu().numpy()[-1]
    classes = [idx_to_class[idx] for idx in top_class.cpu().numpy()[-1]]
    
    return probs, classes

network = Network.load(args.checkpoint)

probs, classes = predict(args.input, network, args.topk)

if args.categories:
    # load category mapping from Json file
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]
    
print(f"Top {args.topk} category matches:")

for p,c in zip(probs, classes):
    print(f"{c} - {(p * 100):.1f}%")