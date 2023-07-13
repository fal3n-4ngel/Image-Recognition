import torch
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


batch_size = 50

dataTransform = transforms.ToTensor()
trainData = FashionMNIST(root='./data', train=True, download=True, transform=dataTransform)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
testData = FashionMNIST(root='./data', train=False,  download=True, transform=dataTransform)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=True)


classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,16,3, padding=1)
        self.fc1 = nn.Linear(7*7*16,75)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(75,10)
       
      
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


net = Net()


modelDirectory = 'saved/'

modelName = input("\nEnter the Model Name (default savedModel):")
if modelName=='':
    modelName="savedModel"
modelName+=".pt"

net.load_state_dict(torch.load( modelDirectory+modelName))


test_loss = torch.zeros(1)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

net.eval()


criterion = nn.CrossEntropyLoss()                                      
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



correct = 0
total = 0

for batch_i, data in enumerate(testLoader):
    
    
    inputs, labels = data
   
    with torch.no_grad():            

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
        _, predicted = torch.max(outputs.data, 1)
        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

        for i in range(batch_size):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: ' % (classes[i]))

        
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


dataiter = iter(testLoader)
images, labels = next(dataiter)

preds = np.squeeze(net(images).data.max(1, keepdim=True)[1].numpy())
images = images.numpy()

fig = plt.figure(figsize=(30, 6))
for idx in np.arange(batch_size//5*2):
    ax = fig.add_subplot(2, batch_size//5, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx] else "red"))
plt.tight_layout()
plt.show()