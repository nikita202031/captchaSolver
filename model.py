import torch
from torch import nn
from torch.nn import functional as F


class CaptchaSolver(nn.Module):
    def __init__(self, num_classes=36):
        super(CaptchaSolver, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Выравнивание размерности после сверток
        self.fc1 = nn.Linear(1280, 64)
        self.drop1 = nn.Dropout(0.3)

        # LSTM слой для обработки последовательности символов
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2,dropout=0.25, batch_first=True)
        
        # Выходной слой для каждого символа в капче
        self.fc_out = nn.Linear(64, num_classes+1)


    def forward(self, images, targets = None):
        # Применение сверточных и батч-нормализационных слоев
        images = images.unsqueeze(0) # 1 16 75 300
        images = images.permute(1,0,2,3)
        bs,c,h,w = images.size()


        #print(f'bs: {bs}, c: {c}, h: {h}, w: {w}')
        x = F.relu(self.conv1(images))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x) # 1, 64, 18, 75
        x = x.permute(0,3,1,2) # 1, 75, 64, 18
        x = x.view(bs,x.size(1), -1)
        x = self.fc1(x)
        x = self.drop1(x)
        x,_ = self.gru(x)
        x = self.fc_out(x)
        x = x.permute(1,0,2)
        #print(x.size())


        if targets is not None:
            log_softmax_values = F.log_softmax(x,2)
            input_lenghts = torch.full(
                size=(bs,),
                fill_value = log_softmax_values.size(0),
                dtype=torch.int32
            )

            target_lenghts = torch.full(
                size=(bs,),
                fill_value = targets.size(1),
                dtype=torch.int32
            )


            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lenghts, target_lenghts
            )
            return x,loss
        return x


if __name__ == '__main__':
    model = CaptchaSolver(22)
    image = torch.randn((1,80,280))
    x,_ = model(image, torch.rand((1,4)))
    print(x)