import torch
from torchvision import models
from torch.autograd import Variable
import model_utils
from PIL import Image
import numpy as np
import argparse
import json
import train

def get_command_line_args():
    parser = argparse.ArgumentParser()
    #-----Required Arguments----------
    parser.add_argument('input', type=str,
                        help='Image file')
    
    parser.add_argument('checkpoint', type=str,
                        help='Saved model checkpoint')

    #-----Optional Arguments----------
        
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return the top K most likely classes')
    parser.set_defaults(top_k=1)
    
    parser.add_argument('--cat_names', type=str,
                        help='File of category names')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU')
    parser.set_defaults(gpu=False)

    return parser.parse_args()

def main():
    # Get input arguments
    args = get_command_line_args()
    use_gpu = torch.cuda.is_available() and args.gpu
    print("Input file: {}".format(args.input))
    print("Checkpoint file: {}".format(args.checkpoint))
    if args.top_k:
        print("Returning {} most likely classes".format(args.top_k))
    if args.category_names:
        print("Category names file: {}".format(args.cat_names))
    if use_gpu:
        print("Using GPU.")
    else:
        print("Using CPU.")
    
    # Load the checkpoint
    model = train.load_checkpoint(args.checkpoint)
    print("Checkpoint loaded.")
    
    # Move tensors to GPU
    if use_gpu:
        model.cuda()
    
    # Load categories file
    if args.cat_names:
        with open(args.cat_names, 'r') as f:
            categories = json.load(f)
            print("Category names loaded")
        
    # Predict
    print("Processing image")
    probabilities, classes, names = predict(args.input, model, args.top_k)
    
    # Show the results
    # Print results
    
    print("Top {} Classes for '{}':".format(len(classes), args.input))
    if args.cat_names:
            print("{:<30} {}".format("Flower", "Probability"))
            print("------------------------------------------")
        else:
            print("{:<10} {}".format("Class", "Probability"))
            print("----------------------")

    for i in range(0, len(classes)):
        if args.cat_names:
            print("{:<30} {:.2f}".format(categories[names[i]], probabilities[i]))
        else:
            print("{:<10} {:.2f}".format(names[i], probabilities[i]))
         
    
if __name__ == "__main__":
    main()

###Source code###
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    size = 224
    img = Image.open(image)
    width, height = img.size
    
    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
    
        
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    crroped_img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    np_img = np.array(crroped_img)/255.
    np_img = (np_img - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])

    np_img = np_img.transpose((2, 0, 1))
    
    return np_img
#######    
 
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
 # TODO: Implement the code to predict the class from an image file
    ###Source code##
    # https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
     # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = tch.from_numpy(img).float().to(device)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = tch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.cpu().topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_classes = [idx_to_class[class] for lab in top_classes]
    top_flowers = [cat_to_name[idx_to_class[class]] for lab in top_classes]
    return top_probs, top_classes, top_flowers
   
##################

# TODO: Display an image along with the top 5 classes

###Source code##
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
def view_classify(image_path, model):
    
     # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('/')[2]
    title = cat_to_name[flower_num]
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title);
    # Make prediction
    probs, labs, flowers = predict(image_path, model) 
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    
#####

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax