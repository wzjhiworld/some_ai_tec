import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage
import torchvision

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = torchvision.models.mobilenet_v2(pretrained=True, progress=True)
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True)

show = ToPILImage()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])#把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
trainset = tv.datasets.CIFAR10(root='data1/',train = True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=24,shuffle=True,num_workers=0)
testset = tv.datasets.CIFAR10('data1/',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=24,shuffle=True,num_workers=0)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

model = model.to(device)

# 初始化
ema = EMA(model, 0.99)
ema.register()

criterion  = torch.nn.CrossEntropyLoss()
criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001,momentum=0.9)

#训练网络
from torch.autograd  import Variable
for epoch in range(4):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss  = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ema.update()
        running_loss += loss.cpu().item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
print("----------finished training---------")
dataiter = iter(testloader)
images, labels = dataiter.next()
print('实际的label: ',' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images/2 - 0.5)).resize((400,100))#？？？？？
outputs = model(Variable(images).to(device))
_, predicted = torch.max(outputs.cpu().data,1)#返回最大值和其索引
print('预测结果:',' '.join('%5s'%classes[predicted[j].cpu().data] for j in range(4)))
correct = 0
total = 0
ema.apply_shadow()
model.eval()
for data in testloader:
    images, labels = data
    outputs = model(Variable(images).to(device))
    _, predicted = torch.max(outputs.cpu().data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('10000张测试集中的准确率为: %d %%'%(100*correct/total))
