import json
import torch as tch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_loaders(data_dir):
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ]),
        'valid': transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])
                                    ]),
        'test': transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])
                                   ]),
                        }


    dirs = {'train': train_dir,
            'valid': valid_dir,
            'test': test_dir}

    # TODO: Load the datasets with ImageFolder (image_datasets)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x])
                     for x in ['train', 'valid', 'test']} 


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = {x: tch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                  for x in ['train', 'valid', 'test']}
    dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}


    device = tch.device("cuda:0" if tch.cuda.is_available() else "cpu")
     return dataloaders, dataset_size, device, dirs


    with open('cat_to_name.json', 'r') as f:
         cat_to_name = json.load(f)
            
def get_model(arch, hidden_units):
    '''
    Load pretrained model
    '''
    OUTPUT_SIZE = 102
 
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_features
    else:
        raise Exception("Unknown model")

    for param in model.parameters():
        param.requires_grad = False

    output_size = OUTPUT_SIZE

  classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    #freeze weights 
    for param in model.parameters():
    param.requires_grad = False
 

    if 'vgg' in arch:
        # overwrite model
        model.classifier = classifier

    return model

def build_model(arch, hidden_units, learning_rate):
    '''
    Build the pretrained model
    '''
    model = get_model(arch, hidden_units)
    print("Pretrained model retrieved.")
    
    # Set the parameters
    optimizer = optim.Adam(parameters, lr=learning_rate)
    optimizer.zero_grad()
    criterion = nn.NLLLoss()
    return model, optimizer, criterion