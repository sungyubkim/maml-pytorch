import torch
import torch.nn as nn
import torch.nn.functional as F

from dropblock import DropBlock2D

from collections import OrderedDict

class Network(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.device = args.device
        self.n_channel = args.n_channel
        self.layers = nn.ParameterDict(OrderedDict([]))
        for i in range(4):
            # add convolution block
            in_channel = 3 if i==0 else self.n_channel
            self.layers.update(
                OrderedDict([
                    ('conv_{}_weight'.format(i), nn.Parameter(torch.zeros(self.n_channel, in_channel, 3, 3))),
                    ('conv_{}_bias'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                    # ('bn_{}_weight'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                    ('bn_{}_bias'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                ])
            )
        # add fc layer
        self.layers.update(
            OrderedDict([
                ('fc_weight', nn.Parameter(torch.zeros(args.n_way, self.n_channel * 5 * 5))),
                ('fc_bias', nn.Parameter(torch.zeros(args.n_way)))
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

    def forward(self, x, tuned_params=None):

        if tuned_params is None:
            params = OrderedDict([(k,v.clone()) for k,v in self.named_parameters()])
        else:
            params = OrderedDict([])
            for k, v in self.named_parameters():
                if k in tuned_params.keys():
                    params[k] = tuned_params[k].clone()
                else:
                    params[k] = v.clone()

        for i in range(4):
            x = F.conv2d(x,
            weight=params['layers.conv_{}_weight'.format(i)],
            bias=params['layers.conv_{}_bias'.format(i)],
            padding=1)
            bn_weight = torch.ones_like(params['layers.bn_{}_bias'.format(i)])
            x = F.batch_norm(x,
            running_mean=None,
            running_var=None,
            weight=bn_weight,
            bias=params['layers.bn_{}_bias'.format(i)],
            training=True)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = x.view(-1, self.n_channel * 5 * 5)

        x = F.linear(x,
        weight=params['layers.fc_weight'],
        bias=params['layers.fc_bias'])
        
        return x

class DenseNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.growth_rate = args.growth_rate
        self.n_block = args.n_block
        self.block_size = args.block_size
        self.layers = nn.ParameterDict(OrderedDict([]))

        self.drop_block_1 = DropBlock2D(block_size=1, drop_prob=0.1)
        self.drop_block_5 = DropBlock2D(block_size=5, drop_prob=0.1)

        # add init conv block
        self.layers.update(
                OrderedDict([
                    # ('bn_weight', nn.Parameter(torch.zeros(3))),
                    ('bn_bias', nn.Parameter(torch.zeros(3))),
                    ('conv_weight', nn.Parameter(torch.zeros(16, 3, 7, 7))),
                    ('conv_bias', nn.Parameter(torch.zeros(16))),
                ])
            )
        
        # add dense blocks
        start_filter = 16
        for i in range(self.n_block):
            for j in range(self.block_size):
                self.layers.update(OrderedDict([
                    # ('bn_bottleneck_{}_{}_weight'.format(i,j), 
                    # nn.Parameter(torch.zeros(self.growth_rate*j + start_filter))),
                    ('bn_bottleneck_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate*j + start_filter))),
                    ('conv_bottleneck_{}_{}_weight'.format(i,j), 
                    nn.Parameter(torch.zeros(4*self.growth_rate, self.growth_rate*j + start_filter, 1, 1))),
                    ('conv_bottleneck_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(4*self.growth_rate))),
                    # ('bn_{}_{}_weight'.format(i,j), 
                    # nn.Parameter(torch.zeros(4*self.growth_rate))),
                    ('bn_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(4*self.growth_rate))),
                    ('conv_{}_{}_weight'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate, 4*self.growth_rate, 3, 3))),
                    ('conv_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate))),
                ]))
            self.layers.update(OrderedDict([
                # ('bn_transition_{}_weight'.format(i), 
                # nn.Parameter(torch.zeros(self.growth_rate*self.block_size + start_filter))),
                ('bn_transition_{}_bias'.format(i), 
                nn.Parameter(torch.zeros(self.growth_rate*self.block_size + start_filter))),
                ('conv_transition_{}_weight'.format(i),
                nn.Parameter(torch.zeros(int(0.5*(self.growth_rate*self.block_size + start_filter)) ,self.growth_rate*self.block_size + start_filter, 1, 1))),
                ('conv_transition_{}_bias'.format(i),
                nn.Parameter(torch.zeros(int(0.5*(self.growth_rate*self.block_size + start_filter)) ))),
            ]))
            start_filter = int(0.5*(self.growth_rate*self.block_size + start_filter))

        # add fc layer
        self.layers.update(OrderedDict([
            ('fc_weight', nn.Parameter(torch.zeros(args.n_way, start_filter * 6 * 6))),
            ('fc_bias', nn.Parameter(torch.zeros(args.n_way)))
        ]))

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

    def forward(self, x, tuned_params=None):

        if tuned_params is None:
            params = OrderedDict([(k,v.clone()) for k,v in self.named_parameters()])
        else:
            params = OrderedDict([])
            for k, v in self.named_parameters():
                if k in tuned_params.keys():
                    params[k] = tuned_params[k].clone()
                else:
                    params[k] = v.clone()

        # apply init conv block
        bn_weight = torch.ones_like(params['layers.bn_bias'])
        x = F.batch_norm(x,
        running_mean=None,
        running_var=None,
        weight=bn_weight,
        bias=params['layers.bn_bias'],
        training=True)
        x = F.relu(x)
        x = F.conv2d(x,
        weight=params['layers.conv_weight'],
        bias=params['layers.conv_bias'])
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        x = self.drop_block_1(x)

        # apply dense blocks
        for i in range(self.n_block):
            for j in range(self.block_size):
                # apply bottleneck conv
                bn_weight = torch.ones_like(params['layers.bn_bottleneck_{}_{}_bias'.format(i,j)])
                x_cur = F.batch_norm(x,
                running_mean=None,
                running_var=None,
                weight=bn_weight,
                bias=params['layers.bn_bottleneck_{}_{}_bias'.format(i,j)],
                training=True)
                x_cur = F.relu(x_cur)
                x_cur = F.conv2d(x_cur,
                weight=params['layers.conv_bottleneck_{}_{}_weight'.format(i,j)],
                bias=params['layers.conv_bottleneck_{}_{}_bias'.format(i,j)])
                # apply conv
                bn_weight = torch.ones_like(params['layers.bn_{}_{}_bias'.format(i,j)])
                x_cur = F.batch_norm(x_cur,
                running_mean=None,
                running_var=None,
                weight=bn_weight,
                bias=params['layers.bn_{}_{}_bias'.format(i,j)],
                training=True)
                x_cur = F.relu(x_cur)
                x_cur = F.conv2d(x_cur,
                weight=params['layers.conv_{}_{}_weight'.format(i,j)],
                bias=params['layers.conv_{}_{}_bias'.format(i,j)],
                padding=1)
                x = torch.cat((x, x_cur), 1)

            # apply transition conv
            bn_weight = torch.ones_like(params['layers.bn_transition_{}_bias'.format(i)])
            x = F.batch_norm(x,
            running_mean=None,
            running_var=None,
            weight=bn_weight,
            bias=params['layers.bn_transition_{}_bias'.format(i)],
            training=True)
            x = F.relu(x)
            x = F.conv2d(x,
            weight=params['layers.conv_transition_{}_weight'.format(i)],
            bias=params['layers.conv_transition_{}_bias'.format(i)],
            padding=1)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
            x = self.drop_block_5(x)

        x = x.view(-1, x.shape[1] * 6 * 6)

        x = F.linear(x,
        weight=params['layers.fc_weight'],
        bias=params['layers.fc_bias'])
        
        return x

if __name__=='__main__':
    from arguments import parse_args
    args = parse_args()
    net = DenseNet(args).to(args.device)
    from pprint import pprint
    pprint([k for k, v in net.named_parameters()])
    x = torch.rand((1, 3, 84, 84)).to(args.device)
    print(net(x).shape)