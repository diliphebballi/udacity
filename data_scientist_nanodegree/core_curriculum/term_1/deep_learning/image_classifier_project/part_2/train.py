# Part 2 - Building the command line application
#
# Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a 
# pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.
# The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. 
# Train a new network on a data set with train.py
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# - Choose architecture: python train.py data_dir --arch "vgg13"
# - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# - Use GPU for training: python train.py data_dir --gpu

# python train.py --data_directory flowers --save_directory checkpoints --network "vgg16" --learning_rate 0.001 --hidden_units 1024 --epochs 10 --gpu

import os

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

import json

import time

import argparse


def parse_input_arguments():
    parser = argparse.ArgumentParser(description = "Train a deep neural network")
    parser.add_argument('--data_directory', type = str, help = 'Dataset path')
    parser.add_argument('--save_directory', type = str, help = 'Path to save trained model checkpoint')
    parser.add_argument('--network', type = str, default = 'vgg16', choices = ['vgg11', 'vgg13', 'vgg16', 'vgg19'], help = 'Model architecture')
    parser.add_argument('--learning_rate', type = float, default = '0.001', help = 'Learning rate')
    parser.add_argument('--hidden_units', type = int, default = '1024', help = 'Number of hidden units')
    parser.add_argument('--epochs', type = int, default = '10', help = 'Number of epochs')
    parser.add_argument('--gpu', action = "store_true", default = True, help = 'Use GPU if available')

    args = parser.parse_args()
    #print(args)

    return args.data_directory, args.save_directory, args.network, args.learning_rate, args.hidden_units, args.epochs, args.gpu


if __name__ == "__main__":

    data_directory, save_directory, network, learning_rate, hidden_units, epochs, gpu = parse_input_arguments()
    # print(data_directory)
    # print(save_directory)
    # print(network)
    # print(learning_rate)
    # print(hidden_units)
    # print(epochs)
    # print(gpu)
    
    if data_directory == None:
        print('Please insert the dataset path')
        exit()
    else:
        train_directory = data_directory + '/train'
        valid_directory = data_directory + '/valid'
        test_directory = data_directory + '/test'
    
        # print(data_directory)
        # print(train_directory)
        # print(valid_directory)
        # print(test_directory)

        # Get the num of classes from the directory
        number_train_classes = len(os.listdir(train_directory))
        number_valid_classes = len(os.listdir(valid_directory))
        number_test_classes = len(os.listdir(test_directory))

        #print(number_train_classes)
        #print(number_valid_classes)
        #print(number_test_classes)

        if (number_train_classes != number_valid_classes) or (number_train_classes != number_test_classes) or (number_valid_classes != number_test_classes):
            print('Error: number of train, valid test classes is not the same')
            exit()
    
        number_classes = number_train_classes

        degrees_rotation = 30
        size_crop = 224
        size_resize = 256
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
        batch_size = 64

        train_transforms = transforms.Compose([transforms.RandomRotation(degrees_rotation),
                                            transforms.RandomResizedCrop(size_crop),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(normalize_mean, normalize_std)
                                            ])

        valid_transforms = transforms.Compose([transforms.Resize(size_resize), 
                                            transforms.CenterCrop(size_crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize(normalize_mean, normalize_std)
                                            ])

        test_transforms = transforms.Compose([transforms.Resize(size_resize), 
                                            transforms.CenterCrop(size_crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize(normalize_mean, normalize_std)
                                            ])

        # Load the datasets with ImageFolder
        train_data = datasets.ImageFolder(train_directory, transform = train_transforms)
        valid_data = datasets.ImageFolder(valid_directory, transform = valid_transforms)
        test_data = datasets.ImageFolder(test_directory, transform = test_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)

        # Load .json mapping file from category label to category name
        with open('cat_to_name.json', 'r') as f:
            category_label_to_name = json.load(f)
            # print(category_label_to_name)

        model = getattr(torchvision.models, network)(pretrained = True)

        #print(model)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        dropout_probability = 0.5
        in_features = 25088
        out_features = hidden_units

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, out_features)),
                                                ('drop', nn.Dropout(p = dropout_probability)),
                                                ('relu', nn.ReLU()),
                                                ('fc2', nn.Linear(out_features, number_classes)),
                                                ('output', nn.LogSoftmax(dim = 1))
                                                ]))
            
        model.classifier = classifier
        #print(model)

        # Train the network

        if gpu == True:
            # Use GPU if it's available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

        print('Using:', device)

        criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

        model.to(device)

        validation_step = True

        print('Training started')
        start_training_time = time.time()

        for epoch in range(epochs):
            train_loss = 0
            for inputs, labels in train_loader:     
                
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                log_probabilities = model.forward(inputs)
                loss = criterion(log_probabilities, labels)
                loss.backward()
                optimizer.step()

                train_loss = train_loss + loss.item()
            
            print('\nEpoch: {}/{} '.format(epoch + 1, epochs),
                '\n    Training:\n      Loss: {:.4f}  '.format(train_loss / len(train_loader))
                )
                
            if validation_step == True:
                
                valid_loss = 0
                valid_accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        log_probabilities = model.forward(inputs)
                        loss = criterion(log_probabilities, labels)
                
                        valid_loss = valid_loss + loss.item()
                
                        # Calculate accuracy
                        probabilities = torch.exp(log_probabilities)
                        top_probability, top_class = probabilities.topk(1, dim = 1)
                        
                        equals = top_class == labels.view(*top_class.shape)
                        
                        valid_accuracy = valid_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
                
                model.train()
            
                print("\n    Validation:\n      Loss: {:.4f}  ".format(valid_loss / len(valid_loader)),
                    "Accuracy: {:.4f}".format(valid_accuracy / len(valid_loader)))
                
        end_training_time = time.time()
        print('Training ended')

        training_time = end_training_time - start_training_time
        print('\nTraining time: {:.0f}m {:.0f}s'.format(training_time / 60, training_time % 60))

        # Do validation on the test set
        print('Validation on the test set')
        test_loss = 0
        test_accuracy = 0
        model.eval()

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_probabilities = model.forward(inputs)
            loss = criterion(log_probabilities, labels)

            test_loss = test_loss + loss.item()

            # Calculate accuracy
            probabilities = torch.exp(log_probabilities)
            top_probability, top_class = probabilities.topk(1, dim = 1)

            equals = top_class == labels.view(*top_class.shape)

            test_accuracy = test_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()

        print("\nTest:\n  Loss: {:.4f}  ".format(test_loss / len(test_loader)),
            "Accuracy: {:.4f}".format(test_accuracy / len(test_loader)))

        
        # Save the checkpoint 

        model.class_to_idx = train_data.class_to_idx

        checkpoint = {'network': network,
                    'input_size': in_features,
                    'output_size': number_classes,
                    'learning_rate': learning_rate,       
                    'batch_size': batch_size,
                    'classifier' : classifier,
                    'epochs': epochs,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx}

        checkpoint_filename = 'checkpoint.pth'
        save_path = ''

        if save_directory == None:
            save_path = checkpoint_filename
        else:
            save_path = save_directory + '/' + checkpoint_filename

        print('Save the checkpoint in {}'.format(save_path))
                
        torch.save(checkpoint, save_path)
else:
    print('Error: script can not run as imported module')
