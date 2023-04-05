import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
# from sympy.matrices import Matrix, GramSchmidt
'''
It is the model that we use, containing all detailed ablation parts and optional components.
you can use it to conduct model and achieve further ablation experiments.
'''
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)

class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    Here we need to make sure K*d is equal to the X dimension
    '''

    def __init__(self, layers,device,K=2, d=4, bn_decay=0.1, mask=False):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.device=device
        self.FC_q = nn.ModuleList()
        self.FC_k = nn.ModuleList()
        self.FC_v = nn.ModuleList()
        self.FC = nn.ModuleList()


        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D*2, activations=F.relu,bn_decay=bn_decay)

    def forward(self, X, time_ind):
        batch_size_ = X.shape[0]
        X = X.transpose(1,3).contiguous()
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            mask = mask.to(self.device)
            print("mask",mask.shape)
            print("attention",attention.shape)
            attention = torch.where(mask, attention, -2 ** 15 + 1)

        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        X = X.transpose(1,3).contiguous()
        return X

class mlp_temporal(nn.Module):
    def __init__(self,c_in, c_out):
        super(mlp_temporal, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self,x, time_ind, layer):
        x = x.transpose(1, 3).contiguous()
        x = self.mlp(x)
        x = x.transpose(1, 3).contiguous()
        return x

class mlp_temporal_new(nn.Module):
    def __init__(self,c_in, c_out,sharing_vector_dim,device):
        super(mlp_temporal_new, self).__init__()
        init_param = torch.diag(torch.ones(sharing_vector_dim)) + torch.ones([sharing_vector_dim, sharing_vector_dim]) / sharing_vector_dim
        self.W = nn.Parameter(torch.randn( c_in, c_out).to(device),
                              requires_grad=True).to(device)
        self.B = nn.Parameter(torch.randn( c_out).to(device),
                              requires_grad=True).to(device)

    def forward(self,x, time_ind):
        x = x.transpose(1, 3).contiguous()
        W = self.W
        B = self.B
        x = torch.einsum("df, bdnt->bfnt", [W, x]) + B.reshape([1, -1, 1, 1])
        x = x.transpose(1, 3).contiguous()
        return x

class lstm(nn.Module):
    def __init__(self,num_nodes,device,nhid,layers):
        super(lstm, self).__init__()
        self.lstm = nn.ModuleList()
        self.device = device
        print("LSTM use RNN")
        self.lstm = nn.RNN(nhid, nhid,1).to(device)

    def forward(self,x, time_ind):
        B,D,N,T = x.shape
        x = x.permute(3, 0, 2, 1)
        x = torch.reshape(x, ((T, B * N, D)))
        h0 = torch.zeros((1, B * N, D)).to(self.device)
        output, hn = self.lstm(x, h0)
        x = torch.reshape(output, ((T, B, N, D))).permute(1, 3, 2, 0)
        return x

class tcn_temporal(nn.Module):
    def __init__(self,residual_channels,dilation_channels,kernel_size,layers,new_dilation):
        super(tcn_temporal, self).__init__()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.filter_convs=nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,kernel_size=(1,kernel_size),dilation=new_dilation)
        self.gate_convs=nn.Conv1d(in_channels=residual_channels,out_channels=dilation_channels,kernel_size=(1, kernel_size), dilation=new_dilation)

    def forward(self,x, time_ind):
        original_size = x.size(3)
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        x = nn.functional.pad(x,(original_size-x.size(3),0,0,0))
        return x
'''
Different from previous work, the temporal relations are learned through the dynamic graph convolution, which has the same structure as spatial graph convolution.
'''
class temporal_gconv(nn.Module):
    '''
    input [64, 32, 207, 13]
    '''
    def __init__(self, dropout, num_nodes, num_steps, device, sharing_vector_dim, total_layer, sharing_mode, inter_dim=4):
        super(temporal_gconv, self).__init__()
        self.nconv = nconv()
        self.dropout = dropout
        self.timevec1 = nn.Parameter(torch.randn(num_steps, inter_dim).to(device),
                                     requires_grad=True).to(device)
        self.timevec2 = nn.Parameter(torch.randn(num_steps, inter_dim).to(device),
                                     requires_grad=True).to(device)
        self.nodevec = nn.Parameter(torch.randn(288, inter_dim).to(device),
                                    requires_grad=True).to(device)
        self.k = nn.Parameter(torch.randn(inter_dim, inter_dim, inter_dim).to(device),
                              requires_grad=True).to(device)

    def forward(self, x, time_ind):
        x = x.transpose(2, 3).contiguous()  # B F N T -> B F T N
        timevec1 = self.timevec1
        timevec2 = self.timevec2
        nodevec = self.nodevec
        k = self.k
        adp1 = torch.einsum("ad, def->aef", [nodevec[time_ind], k])
        adp2 = torch.einsum("be, aef->abf", [timevec1, adp1])
        adp3 = torch.einsum("cf, abf->abc", [timevec2, adp2])
        adp = F.softmax(F.relu(adp3), dim=2)
        # x1 = torch.einsum('ncvl,lvw->ncwl', (x, adp)).contiguous()
        x1 = self.nconv(x, adp)
        h = F.dropout(x1, self.dropout, training=self.training)
        h = h.transpose(2, 3).contiguous()
        return h

'''
this module corresponding the dynamic spatial graph convolution
if conduct ablation study "w/o dygraph", we replace the "adp4" with "static_adp0" in the forward fuction.  
'''
class spatial_gconv(nn.Module):
    '''
    input [64, 32, 207, 13]
    '''

    def __init__(self, dropout, num_nodes, device, sharing_vector_dim, total_layer, sharing_mode, inter_dim=4):
        super(spatial_gconv, self).__init__()
        self.nconv = nconv()
        self.dropout = dropout
        self.static_nodevec1 = nn.Parameter(torch.randn(num_nodes, inter_dim).to(device),
                                            requires_grad=True).to(device)
        self.static_nodevec2 = nn.Parameter(torch.randn( num_nodes, inter_dim).to(device),
                                            requires_grad=True).to(device)
        self.nodevec1 = nn.Parameter(torch.randn( num_nodes, inter_dim).to(device),
                                     requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn( num_nodes, inter_dim).to(device),
                                     requires_grad=True).to(device)
        self.timevec = nn.Parameter(torch.randn( 288, inter_dim).to(device), requires_grad=True).to(
            device)
        self.k = nn.Parameter(torch.randn(inter_dim, inter_dim, inter_dim).to(device),
                              requires_grad=True).to(device)
        print("spatial dynamic graph only")


    def forward(self, x, time_ind):
        # static_nodevec1 = torch.einsum("l, lnd->nd", self.co_static_nodevec1[layer], self.static_nodevec1)
        # static_nodevec2 = torch.einsum("l, lnd->nd", self.co_static_nodevec2[layer], self.static_nodevec2)
        nodevec1 = self.nodevec1 #[num_nodes, inter_dim]
        nodevec2 = self.nodevec2 #[num_nodes, inter_dim]
        timevec = self.timevec #[288, inter_dim]
        k = self.k #[inter_dim, inter_dim, inter_dim]

        adp1 = torch.einsum("ad, def->aef", [timevec[time_ind], k])
        adp2 = torch.einsum("be, aef->abf", [nodevec1, adp1])
        adp3 = torch.einsum("cf, abf->abc", [nodevec2, adp2])
        adp4 = F.softmax(F.relu(adp3), dim=2)
        adp = adp4
        x1 = self.nconv(x, adp)
        h = F.dropout(x1, self.dropout, training=self.training)
        return h

class feature(nn.Module):#这个和linear 是一样的
    def __init__(self, c_in, c_out):
        super(feature, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class GDGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256,
                 kernel_size=2, layers=8, spatial_dim=10, temporal_dim=4, temporal_mode = 'node_specific', ablation_mode = 'none'):
        super(GDGCN, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.residual_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.result_fuse = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.temporal_mode = temporal_mode

        sharing_mode = 'a'
        sharing_vector_dim = layers #if mode is a，then sharing_vector_dim must equal to layer
        print("layers: ", layers)
        print("sharing_vector_dim: ", sharing_vector_dim)
        print("spatial_dim: ", spatial_dim)
        print("temporal_dim: ", temporal_dim)
        print("temporal_mode::::",temporal_mode)
        print("sharing_mode::::",sharing_mode)

        self.spatial = spatial_gconv(dropout, num_nodes, device, sharing_vector_dim, layers, sharing_mode, spatial_dim)

        if temporal_mode == 'tcn':
            print("tcn")
            self.temporal = tcn_temporal(residual_channels,dilation_channels,kernel_size,layers,new_dilation=1)
        elif temporal_mode == 'lstm':
            print("lstm")
            self.temporal = lstm(num_nodes,device,dilation_channels,layers)
        elif temporal_mode == 'attention':
            print("attention")
            self.temporal = temporalAttention(layers,device=device)
        elif temporal_mode == 'mlp':
            print("MLP temporal")
            self.temporal = mlp_temporal(12,12)
        elif temporal_mode == 'mlp_new':
            print("MLP temporal new")
            self.temporal = mlp_temporal_new(12, 12,sharing_vector_dim,device)
        else:
            print("node_specific temporal")
            self.temporal = temporal_gconv(dropout, num_nodes, 12, device, sharing_vector_dim, layers, sharing_mode, temporal_dim)

        self.feature = feature(dilation_channels, residual_channels)

        self.layer_spatial = nn.ModuleList()
        self.layer_temporal = nn.ModuleList()
        self.layer_feature = nn.ModuleList()
        for b in range(layers):
            if temporal_mode == 'tcn':
                print("tcn")
                temporal_module = tcn_temporal(residual_channels, dilation_channels, kernel_size, layers, new_dilation=1)
            elif temporal_mode == 'lstm':
                print("lstm")
                temporal_module = lstm(num_nodes, device, dilation_channels, layers)
            elif temporal_mode == 'attention':
                print("attention")
                temporal_module = temporalAttention(layers, device=device)
            elif temporal_mode == 'mlp':
                print("MLP temporal")
                temporal_module = mlp_temporal(12, 12)
            elif temporal_mode == 'mlp_new':
                print("MLP temporal new")
                temporal_module = mlp_temporal_new(12, 12, sharing_vector_dim, device)
            else:
                print("node_specific temporal")
                temporal_module = temporal_gconv(dropout, num_nodes, 12, device, sharing_vector_dim, layers, sharing_mode,
                                               temporal_dim)

            self.layer_spatial.append(spatial_gconv(dropout, num_nodes, device, sharing_vector_dim, layers, sharing_mode, spatial_dim))
            self.layer_temporal.append(temporal_module)
            self.layer_feature.append(feature(dilation_channels, residual_channels))

            self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            if temporal_mode == 'no_temporal':
                self.result_fuse.append(
                    torch.nn.Conv2d(dilation_channels * 4, residual_channels, kernel_size=(1, 1), padding=(0, 0),
                                    stride=(1, 1), bias=True))
            else:
                self.result_fuse.append(
                torch.nn.Conv2d(dilation_channels * 6, residual_channels, kernel_size=(1, 1), padding=(0, 0),
                                stride=(1, 1), bias=True))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=out_dim * residual_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, input, time_ind):
        x = input
        x = self.start_conv(x)
        skip = 0
        for i in range(self.layers):
            residual = x
            spatial_a = self.spatial(x, time_ind)
            temporal_a = self.temporal(x, time_ind)
            feature_a = self.feature(x)
            spatial_b = self.layer_spatial[i](x,time_ind)
            temporal_b = self.layer_temporal[i](x,time_ind)
            feature_b = self.layer_feature[i](x)


            if self.temporal_mode == 'no_temporal':
                x = torch.cat([spatial_a, feature_a, spatial_b, feature_b], dim=1)
            else:
                x = torch.cat([spatial_a, spatial_b, temporal_a, temporal_b, feature_a, feature_b], dim=1)

            x = F.relu(x)
            x = self.result_fuse[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = torch.transpose(x, 3, 2)
        x = torch.reshape(x, (x.size(0), x.size(1) * x.size(2), x.size(3), 1))
        x = self.end_conv_2(x)
        return x
