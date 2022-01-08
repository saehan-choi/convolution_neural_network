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

optimizer = optim.Adam(model.parameters(), lr=0.001)
# parameters 말고 parameters()임.
criterion = nn.CrossEntropyLoss().cuda()



learning_rate = [0.001, 0.005, 0.0001]
batch_size = [30, 40, 50]

for lr in learning_rate:
  for b_size in batch_size:
    hyper_parameters = {
        'learning_rate' : lr,
        'epochs' : 10,
        'batch_size' : b_size,
        'storage_location' : 'C:\m_results\holy.txt'
    }
    # 이거 조심해야하는게, 실험시마다 text name 을 변경해줘야함 항상 넣어주지 않으면 겹쳐져서 헷갈림,......

    f = open(hyper_parameters['storage_location'], 'a')
    # 기존파일에 내용 추가하고싶을경우 'w' 가아닌 'a'로 해야함

    train_data = ImageFolder("./train", 
                                transform=transforms.Compose
                                ([transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])
                            )
    train_loader = DataLoader(train_data, batch_size=hyper_parameters['batch_size'], shuffle=True)

    for epoch in range(hyper_parameters['epochs']):
        train_loss = 0.0
        train_correct = 0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            outputs = model(inputs.to("cuda"))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # train_correct += (predicted == labels).cpu().sum()
        print(f'train_loss {train_loss} epoch {epoch} learning_rate {lr} batch_size {b_size}')
        f.write(f'train_loss {train_loss} epoch {epoch} learning_rate {lr} batch_size {b_size}\n')