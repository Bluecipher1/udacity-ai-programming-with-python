import torch
from torchvision import datasets, transforms
import argparse
from my_module import Network

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--arch", dest="arch", default='vgg13')
parser.add_argument("--save_dir", dest="checkpoint", default="checkpoint.pth")
parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.002)
parser.add_argument("--hidden_units", dest="hidden_units", default=[8192, 1024])
parser.add_argument("--dropout", dest="drop_p", type=float, default=0.5)
parser.add_argument("--epochs", dest="epochs", type=int, default=5)
parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)

args = parser.parse_args()

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms) 
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms) 

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)

network = Network(arch=args.arch, hidden_units=args.hidden_units, output_size=102, learning_rate=args.learning_rate,
                 drop_p=args.drop_p, gpu=args.gpu)

network.train(trainloader, validationloader, train_data.class_to_idx, train_data.class_to_idx, args.epochs)

network.save(args.checkpoint)

