import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict

class Network():
    """ Initializes a neural network for image classification. It uses a pretrained
    model from torchvision and modifies it with a custom clssifier defined by the parameters
    as specified below.
    
    Parameters:
        arch (string) - Torchvision model arcitecture, e.g. 'vgg13'
        hidden_unit (list of ints) - specifies the sizes of the classifier's hidden layers
        output_size (int) - size of the outlayer, i.e. number of categories
        drop_p (float) - Dropout probablity for the hidden layers
        learning_rate (float) - the learning_rate
        gpu (bool) - use the GPU for training (if available)
    """
    def __init__(self, arch, hidden_units, output_size, drop_p, learning_rate, gpu):

        # load specified torchvision model
        model_to_call = getattr(models, arch)
        self.model = model_to_call(pretrained=True)

        # freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
        # get last direct child of model, which is assumed to be the classifier
        for name, module in self.model.named_children():
            pass

        # get input size of classifier
        input_size = next(iter(module.children())).in_features

        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        
        layers = OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_units[0])),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2))
        ])
        
        i = 2
        for h1, h2 in layer_sizes:
            layers.update([
                (f"fc{i}", nn.Linear(h1, h2)),
                (f"relu{i}", nn.ReLU()),
                (f"dropout{i}", nn.Dropout(p=drop_p))
            ])
            i += 1
            
        layers.update([
            (f"fc{i}", nn.Linear(hidden_units[-1], output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ])
        
        classifier = nn.Sequential(layers)
        # replace model classifier with our version
        setattr(self.model, name, classifier)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

        self.arch = arch
        self.hidden_units = hidden_units
        self.epochs = 0
        self.output_size = output_size
        self.drop_p = drop_p
        self.learning_rate = learning_rate
        self.gpu = gpu

        self.model.to(self.device)

    def train(self, trainloader, validationloader, class_to_idx, epochs=5):
        """ Trains the neural network using the images provided by the trainloader
        and validating its accuracy using the validationloader.
    
        Parameters:
            trainloader (DataLoader) - Provides the images to train on
            trainloader (DataLoader) - Provides the images to validate against
            epochs (int) - number of epochs to iterate
        """
        steps = 0
        print_every = 5

        train_loss = 0

        self.class_to_idx = class_to_idx
        total_epochs = self.epochs + epochs
        
        for epoch in range(self.epochs, total_epochs):
    
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_ps = self.model.forward(images)
                loss = self.criterion(log_ps, labels)
                loss.backward()
                self.optimizer.step()
        
                train_loss += loss.item()
        
                if steps % print_every == 0:
                    self.model.eval()
            
                    valid_loss = 0
                    accuracy = 0
            
                    with torch.no_grad():
                        for images, labels in validationloader:
                            images, labels = images.to(self.device), labels.to(self.device)
                            log_ps = self.model.forward(images)
                            valid_loss += self.criterion(log_ps, labels).item()

                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                    print(f"Epoch {epoch+1}/{total_epochs}.. "
                          f"Train loss: {train_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validationloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            
                    train_loss = 0
                    self.model.train()
                    
        self.epochs += epochs
        
    def predict(self, data, topk=5):
        """ Predicts probabilities and classes for specified data.
        
        Parameters:
            data - tensor with images
            topk - number of top matche to return
        
        Returns:
            (probs, classes) - tuple with two lists
        """
        self.model.eval()
        images = data.to(self.device)
        
        with torch.no_grad():
            log_ps = self.model.forward(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(topk, dim=1)
        self.model.train()
        
        return top_p, top_class

    def save(self, filepath):
        """ Saves the current state of the network to a checkpoint file.
        
        Parameters:
            filepath (string) - path of the checkpoint file, e.g. 'checkpoint.pth'
        """
        checkpoint = {
            'arch': self.arch,
            'hidden_units': self.hidden_units, 
            'output_size': self.output_size,
            'drop_p': self.drop_p,
            'learning_rate': self.learning_rate,
            'gpu': self.gpu,
            'epochs': self.epochs,
            'optimizer_state': self.optimizer.state_dict(),
            'model_state': self.model.state_dict(),
            'class_to_idx': self.class_to_idx
        }

        torch.save(checkpoint, filepath)
        
    @staticmethod
    def load(filepath):
        """ Loads a model from the specified checkpoint file
        
        Parameters:
            filepath - file name of the checkpoint file
            
        Returns:
            initialized Network model
        """
        checkpoint = torch.load(filepath)
        
        network = Network(
            arch = checkpoint['arch'],
            hidden_units = checkpoint['hidden_units'],
            output_size = checkpoint['output_size'],
            drop_p = checkpoint['drop_p'],
            learning_rate = checkpoint['learning_rate'],
            gpu = checkpoint['gpu']
        )
        network.epochs = checkpoint['epochs']
        network.class_to_idx = checkpoint['class_to_idx']
        network.optimizer.load_state_dict(checkpoint['optimizer_state'])
        network.model.load_state_dict(checkpoint['model_state'])
        
        return network