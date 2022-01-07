import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import time
# import numpy as np

# from PIL import Image

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = ImageFolder("./train/",
                         transform=transform)
trainloader = data.DataLoader(trainset, batch_size=4 , shuffle=True)


# testset = ImageFolder("./asl_alphabet_test/",
#                          transform=transform)
# testloader = data.DataLoader(testset, batch_size=4, shuffle=True)

# for data in testloader:
#     images, labels = data
#     images = images.cuda()
#     labels = labels.cuda()
#     print(labels)
class VGG_E(nn.Module):
    def __init__(self, num_classes: int = 4, init_weights: bool = True):
        super(VGG_E, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3x64 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv64x64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
        )
        self.conv64x128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
        )
        self.conv128x128 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
        )
        self.conv128x256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
        )
        self.conv256x256 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
        )        
        self.conv256x512 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
        )
        self.conv512x512 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
        )
        self.convnet = nn.Sequential(
            # Input Channel (RGB: 3)
            self.conv3x64,

            self.conv64x64,
            self.conv64x64,
            # self.conv64x64,
            # self.conv64x64,        
            # self.conv64x64,
            # self.conv64x64,                  

           
            self.conv64x128,
            self.maxpool, # 224 -> 112 

            self.conv128x128,
            self.conv128x128,
            # self.conv128x128,
            # self.conv128x128,            
            # self.conv128x128,
            # self.conv128x128,   

            self.conv128x256,
            self.maxpool, # 112 -> 56

            self.conv256x256,
            self.conv256x256,
            # self.conv256x256,
            # self.conv256x256,            
            # self.conv256x256,
            # self.conv256x256,
            # self.conv256x256,
            # self.conv256x256,    
            # self.conv256x256,
            # self.conv256x256,
            # self.conv256x256,

            self.conv256x512,


            
            self.maxpool, # 56 -> 28
            
            self.conv512x512,
            self.conv512x512,
            # self.conv512x512,
            # self.conv512x512,            
            # self.conv512x512,            
            self.maxpool # 28 -> 14

        )

        self.fclayer = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 2),
            # nn.Softmax(dim=1), # Loss인 Cross Entropy Loss 에서 softmax를 포함한다.
        )
    
    def make_conv(self,in_dim,mid_dim,out_dim=False):
        nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=mid_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )
        return nn.Sequential




    def forward(self, x:torch.Tensor):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        # print(x.shape)
        return x




device = torch.device('cuda')

vgg11 = VGG_E()
# vgg11 = VGG_E(num_classes = 4)
vgg11 = vgg11.to(device)




criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(vgg11.parameters(),lr=0.00005)
start_time = time.time()

for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # print(inputs.shape)  
        outputs= vgg11(inputs)
        print(labels)
        print(outputs)
        # print('this is output shape')  64
        # print(outputs.shape)
        # print('this is label shape')
        # print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

###########################################
PATH='./plainnet_weight/' 
# 지금 epochs 20 으로했는데 10으로 한것도 다른이름으로 저장해서
# 과적합이 맞는지 찾은다음 더 좋은방안을 찾자
torch.save(vgg11.state_dict(), PATH+'model.pt')
###########################################
print(time.time()-start_time)


print('Finished Training')

# vgg11=vgg11() 새로 정의할필요가없음 새로정의하면 가중치가 제대로 저장안됨..ㅋㅋ
# vgg11.load_state_dict(torch.load(PATH+'model.pt'))
# 이부분은 내가 추가했음 필요없을지도 모름.
class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))
classes =  ('A', 'B', 'C', 'D','del', 'E', 'F','nothing')

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()


        outputs = vgg11(images)
        _, predicted = torch.max(outputs, 1)

        print('this is predicted')
        print(predicted)
        

        c = (predicted == labels).squeeze()

        print('this is c = (predicted == labels).squeeze()')
        print(c)



        for i in range(8):
            # 여기 64개는 미니배치수대로 하면됨 ㅋㅋㅋㅋ 안되면 실험해봐라
            # 이거 테스트데이터 갯수에 맞춰야함...!!!!!!!!
            label = labels[i]
            print('what is label?')
            print(label)

            class_correct[label] += c[i].item()
            print('what is class_correct[label]?')
            print(class_correct[label])
            print('what is c[i].item()')
            print(c[i].item())
            class_total[label] += 1
            print('what is class_total[label]')
            print(class_total[label])
            print('\n')
            print('\n')