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


def train(epochNum):
    
    training_loss=[]
    
    for epoch in range(epochNum):  

        running_loss = 0.0
        for batch_i, data in enumerate(trainLoader):
            
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  
            if batch_i % 1000 == 999:  
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                training_loss.append(running_loss/1000) 
                running_loss = 0.0

    print('Finished Training')
    return training_loss


batch_size = 50

dataTransform = transforms.ToTensor()
trainData = FashionMNIST(root='./data', train=True, download=True, transform=dataTransform)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
testData = FashionMNIST(root='./data', train=False,  download=True, transform=dataTransform)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=True)

print('\nNumber of Training Images: ', len(trainData))
print('Test data, number of images: ', len(testData))

classes = [ 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot','T-shirt/top', 'Trouser', 'Pullover', 'Dress']

dataIterater = iter(trainLoader)
images, labels = next(dataIterater)
images = images.numpy()

fig = plt.figure(figsize=(20,6))
for idx in np.arange(batch_size//3*2):
    ax = fig.add_subplot(2, batch_size//3, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])


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


criterion = nn.CrossEntropyLoss()                                      
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


correct = 0
total = 0


for images, labels in testLoader:
    
    images = Variable(images)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()           

accuracy = 100 * correct / total
print('\nAccuracy before training: ', accuracy)



epochNum = int(input("\nEnter the epoch Number: (1-15000):"))





training_loss = train(epochNum)

plt.plot(training_loss)
plt.show()


# Saving the model 
modelDirectory = 'saved/'
modelName = input("\nEnter the Model Name (default savedModel):")
if modelName=='':
    modelName="savedModel"
modelName+=".pt"
torch.save(net.state_dict(), modelDirectory+modelName)