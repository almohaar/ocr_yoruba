import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        """
        imgH: Image height (must be a multiple of 16)
        nc: Number of channels (3 for RGB)
        nclass: Number of classes (including the CTC blank)
        nh: Number of hidden units in LSTM layers
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "Image height must be a multiple of 16."
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 1/2 size
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 1/4 size
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), padding=(0,1)),  # 1/8 size
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), padding=(0,1)),  # 1/16 size
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, x):
        conv = self.cnn(x)  # [B, c, H, W]
        b, c, h, w = conv.size()
        assert h == 1, "The height of conv must be 1."
        conv = conv.squeeze(2)  # [B, c, W]
        conv = conv.permute(2, 0, 1)  # [W, B, c]
        output = self.rnn(conv)  # [W, B, nclass]
        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        recurrent = recurrent.view(T * b, h)
        output = self.embedding(recurrent)
        output = output.view(T, b, -1)
        return output
