import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d

# ResNet 모델을 구현한다
class ResModel(torch.nn.Module):
    def __init__(self):
        super(ResModel, self).__init__()
        # 이번 경진대회에 사용되는 label 개수는 12이다
        n_labels = 12
        n_maps = 128
        # 총 9계층 모델을 쌓는다
        self.n_layers = n_layers = 9
        # 첫 계층에 사용하는 convolutional 모듈을 정의한다
        self.conv0 = torch.nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        # MaxPooling 모듈을 정의한다
        self.pool = MaxPool2d(2, return_indices=True)
        # 2계층 이후에 사용하는 convolutional 모듈을 정의한다
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False) for _ in range(n_layers)])
        # BatchNormalization 모듈과 conv 모듈을 조합한다
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), torch.nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        # 최종 계층에는 선형 모듈을 추가한다
        self.output = torch.nn.Linear(n_maps, n_labels)

    def forward(self, x):
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                old_x = y
            # 이전 layer의 결과값(old_x)와 이번 layer 결과값(y)을 더하는 것이 residual 모듈이다 
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            # BatchNormalization을 통해 파라미터 값을 정규화한다
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
            # pooling을 사용할지 True/False로 지정한다
            pooling = False
            if pooling:
                x_pool, pool_indices = self.pool(x)
                x = self.unpool(x_pool, pool_indices, output_size=x.size())
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        # 최종 선형 계층을 통과한 결과값을 반환한다
        return self.output(x)
