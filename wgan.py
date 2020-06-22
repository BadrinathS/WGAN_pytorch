#WGAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import wandb

wandb.init(job_type='train', project='WGAN', name='WGAN')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 500

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_dataset = datasets.FashionMNIST(root = './fashion_mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root= './fashion_mnist_data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)

        return x
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x)             # No sigmoid

        return x

g_input_dim = 100
g_output_dim = train_dataset.train_data.size(1)*train_dataset.train_data.size(2)

d_input_dim = train_dataset.train_data.size(1)*train_dataset.train_data.size(2)


generator = Generator(g_input_dim, g_output_dim).to(device)
discriminator = Discriminator(d_input_dim).to(device)

criterion = nn.BCELoss()

lr = 0.0002
generator_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)

def train_discriminator(x):
    discriminator.zero_grad()

    x_real, y_real = x.view(-1,g_output_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    discriminator_real_output = discriminator(x_real)
    # discriminator_real_loss = criterion(discriminator_output, y_real)
    # discriminator_real_loss = torch.log(discriminator_output + 0.0001)

    z = torch.randn(bs, g_input_dim).to(device)
    x_fake, y_fake = generator(z), torch.zeros(bs, 1).to(device)

    discriminator_fake_output = discriminator(x_fake)
    # discriminator_fake_loss = criterion(discriminator_output, y_fake)
    # discriminator_fake_loss = torch.log(1-discriminator_output + 0.0001)

    discriminator_loss = (torch.mean(discriminator_real_output) - torch.mean(discriminator_fake_output))
    discriminator_loss.backward()
    discriminator_optimizer.step()

    for p in discriminator.parameters():
        p.data.clamp_(-0.001,0.001)
    

    return discriminator_loss.item()

def train_generator(x):
    generator.zero_grad()

    z = torch.randn(bs, g_input_dim).to(device)
    y = torch.ones(bs, 1).to(device)

    generator_output = generator(z)
    discriminator_output = discriminator(generator_output)
    # generator_loss = criterion(discriminator_output, y)
    # generator_loss = -torch.mean(torch.log(discriminator_output + 0.0001)) #Use either of loss function, although this one is easier to optimize
    generator_loss = torch.mean(discriminator_output)


    generator_loss.backward()
    generator_optimizer.step()

    return generator_loss.item()

n_epoch = 200
for epoch in range(n_epoch):

    d_losses, g_losses = [], []


    for batch_id, (x,y) in enumerate(train_loader):
        g_losses.append(train_generator(x))
        d_losses.append(train_discriminator(x))

        wandb.log({'Generator Loss':g_losses[-1], 'Discriminator Loss':d_losses[-1]}, step=epoch)
    
    if epoch % 10 == 0:
        with torch.no_grad():
            test_z = torch.randn(bs, g_input_dim).to(device)
            generated_output = generator(test_z)
            name = './images/sample_' + str(epoch) + '.png'
            save_image(generated_output.view(bs, 1, 28,28), name)
        
    print('Epoch %d \t loss_d %f \t loss_g %f'%( epoch, torch.mean(torch.FloatTensor(d_losses)).item(), torch.mean(torch.FloatTensor(g_losses)).item()))

torch.save(generator.state_dict(), './ckpt/generator.pth')
torch.save(discriminator.state_dict(), './ckpt/discriminator.pth')

if epoch % 10 == 0:
    with torch.no_grad():
        test_z = torch.randn(bs, g_input_dim).to(device)
        generated_output = generator(test_z)
        name = './images/sample_final.png'
        save_image(generated_output.view(bs, 1, 28,28), name)