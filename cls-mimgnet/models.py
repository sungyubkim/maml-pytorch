import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class Network(nn.Module):
    def __init__(self, n_channel=32, n_way=5):
        super().__init__()

        self.n_channel = n_channel
        self.layers = nn.ParameterDict(OrderedDict([]))
        for i in range(4):
            # add convolution block
            in_channel = 3 if i==0 else self.n_channel
            self.layers.update(
                OrderedDict([
                    ('conv_{}_weight'.format(i), nn.Parameter(torch.zeros(self.n_channel, in_channel, 3, 3))),
                    ('conv_{}_bias'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                    ('bn_{}_weight'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                    ('bn_{}_bias'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                ])
            )
        # add fc layer
        self.layers.update(
            OrderedDict([
                ('fc_weight', nn.Parameter(torch.zeros(n_way, self.n_channel * 5 * 5))),
                ('fc_bias', nn.Parameter(torch.zeros(n_way)))
            ])
        )

        self.init_params()

    def init_params(self):

        for k, v in self.named_parameters():
            if ('conv' in k) or ('fc' in k):
                if ('weight' in k):
                    nn.init.xavier_normal_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('bn' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)

    def forward(self, x, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        for i in range(4):
            x = F.conv2d(x,
            weight=params['layers.conv_{}_weight'.format(i)],
            bias=params['layers.conv_{}_bias'.format(i)],
            padding=1)
            x_reshape = x.permute(1, 0, 2, 3).contiguous().detach() # (C, N, H, W)
            x_reshape = x_reshape.view(self.n_channel, -1) # (C, N * H * W)
            running_mean = x_reshape.mean(1) # (C)
            running_var = x_reshape.var(1) # (C)
            x = F.batch_norm(x,
            running_mean=running_mean,
            running_var=running_var,
            weight=params['layers.bn_{}_weight'.format(i)],
            bias=params['layers.bn_{}_bias'.format(i)],
            momentum=1.0,
            training=True)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = x.view(-1, self.n_channel * 5 * 5)

        x = F.linear(x,
        weight=params['layers.fc_weight'],
        bias=params['layers.fc_bias'])
        
        return x

if __name__=='__main__':
    net = Network()
    from pprint import pprint
    pprint([k for k, v in net.named_parameters()])
    x = torch.rand((1, 3, 84, 84))
    print(net(x).shape)