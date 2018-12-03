import pretrain
import numpy as np
import torch as tch
from torch import nn
from torch import optim
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import os
import argparse

def get_command_line_args():
    parser = argparse.ArgumentParser()
    #-----Required Arguments----------
    parser.add_argument('data_dir', type=str,
                        help='Directory of flower images')
    
    #-----Optional Arguments----------
    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Train with GPU')
    parser.set_defaults(gpu=False)
             
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save checkpoints')
    
    parser.add_argument('--arch', dest='arch', default='vgg19', action='store',
                        help='Architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Model learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--output', type=int, default=102,
                        help='Number of classes to classify')
    return parser.parse_args()

def main():
    
    # Get Command Line Arguments
    args = get_command_line_args()
    use_gpu = torch.cuda.is_available() and args.gpu
    print("Data directory: {}".format(args.data_dir))
    if use_gpu:
        print("Training on GPU.")
    else:
        print("Training on CPU.")
    print("Architecture: {}".format(args.arch))
    if args.save_dir:
        print("Checkpoint save directory: {}".format(args.save_dir))
        
    print("Learning rate: {}".format(args.learning_rate))
    print("Hidden units: {}".format(args.hidden_units))
    print("Epochs: {}".format(args.epochs))
    print("Output classes: {}".format(args.output))
    # Get data loaders
    dataloaders, class_to_idx = model_utils.get_loaders(args.data_dir)
    for key, value in dataloaders.items():
        print("{} data loader retrieved".format(key))
    
    # Build the model
    model, optimizer, criterion = pretrain.build_model(args.arch, args.hidden_units, args.learning_rate)
    model.class_to_idx = class_to_idx
    
    # Check if GPU availiable and move
    if use_gpu:
        print("GPU is availaible. Moving Tensors.")
        model.cuda()
        criterion.cuda()
    
    # Train the model
    
    train(model, args.epochs, criterion, optimizer, dataloaders)
    
    # Save the checkpoint
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = args.save_dir + '/' + args.arch + '_checkpoint.pth'
    else:
        save_path = args.arch + '_checkpoint.pth'
    print("Will save checkpoint to {}".format(save_path))

    save_checkpoint(model, args.arch, args.output, save_path)
    print("Checkpoint saved")

    # Validate the accuracy validate(model, criterion, data_loader)
    test_loss, accuracy = validate(model, criterion, dataloaders['test'])
    print("Test Loss: {:.3f}".format(test_loss))
    print("Test Acc.: {:.3f}".format(accuracy))
          
if __name__ == "__main__":
    main()

       
def validate(model, criterion, data_loader):
    model.eval()
    accuracy = 0
    test_loss = 0
        
    with tch.no_grad():
        for inputs, labels in iter(data_loader):
            if tch.cuda.is_available():
                inputs = Variable(inputs.float().cuda(), volatile=True)
                labels = Variable(labels.long().cuda(), volatile=True) 
            else:
                inputs = Variable(inputs, volatile=True)
                labels = Variable(labels, volatile=True)

            output = model.forward(inputs)
            test_loss += criterion(output, labels).data[0]
            ps = tch.exp(output).data 
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(tch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)
    
#training data
def train(model, epochs, criterion, optimizer, dataloader):
    
    print("Beginning model training")
    print_every = 40
    steps = 0
    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloader['train']):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss, accuracy = validate(model, criterion, dataloader['valid'])
               
                print("Epoch: {}/{}... ".format(e+1, epochs))
                print('-'* 10)
                print("Train Loss: {:.4f}".format(running_loss/print_every))
                print("Validation Loss: {:.4f} ".format(validation_loss),
                      "Validation Accuracy: {:.4f}".format(accuracy))
                print()
                running_loss = 0

#saving checkpoint                
def save_checkpoint(model, arch, output, save_path):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'output': output
                 }

    tch.save(checkpoint, save_path)
    
    
def load_checkpoint(checkpoint_path):
    
    chpt = tch.load(checkpoint_path)
    
    model = models.chpt['arch'](pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = chpt['class_to_idx']
    
    # Create the classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096, chpt['output'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model