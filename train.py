import torch
import torch.nn
import wandb
from resnet import *
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


device = torch.device('cuda')
model = ResNet18(img_channel=3, num_classes=2)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# parameters 말고 parameters()임.
criterion = nn.CrossEntropyLoss().cuda()

train_data = ImageFolder("./train", 
                            transform=transforms.Compose
                            ([transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
                        )
train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
wandb.init(project="my-test-project", entity="saehoni", name='haha')

wandb.config = {
  "learning_rate": 0.0001,
  "epochs": 10,
  "batch_size": 50
}

config = wandb.config

print(config.learning_rate)
# 이거는 제대로 모르겠네 learning rate 랑 epoch를 

# print(f'config_learning_rate:{wandb.config.learning_rate}')
for epoch in range(10):
    train_loss = 0.0
    train_correct = 0
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        outputs = model(inputs.to("cuda"))
        # _, predicted = torch.max(outputs, 1)
        # print(labels)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # train_correct += (predicted == labels).cpu().sum()
    print(f'train_loss : {train_loss}')
    print(f'epoch step :{epoch}')
    wandb.log({"train_loss": train_loss})
    wandb.watch(model)

    # y = model(torch.randn(1, 3, 224, 224)).to("cuda")
    # # print(y[0][1])
    # print(y)
    # print(y.size())