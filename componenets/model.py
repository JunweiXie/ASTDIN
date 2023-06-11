import torch
import torch.nn as nn
from util import reparameterize


import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_dim, num_nodes=None, cut_size=0,device=None):
        super(TemporalAttention, self).__init__()
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.key_proj = LinearCustom()
        self.value_proj = LinearCustom()

        self.projection1 = nn.Linear(in_dim,in_dim)
        self.projection2 = nn.Linear(in_dim,in_dim)


        self.rate = nn.Parameter(torch.zeros(1) + 0.8)

        self.device = device

    def forward(self, query, key, value, parameters):
        batch_size = query.shape[0]
        T = key.shape[1]
        N = key.shape[2]

        time_decays = torch.tensor(range(T - 1, -1, -1), dtype=torch.float16).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(self.device)
        b_time_decays = time_decays.repeat(batch_size * self.K, N, 1, T) + 1

        key = self.key_proj(key, parameters[0])
        value = self.value_proj(value, parameters[1])

        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)


        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)
        attention /= (self.head_size ** 0.5)



        denominator = 1+torch.sigmoid(torch.sigmoid(attention)*torch.sigmoid(self.rate)* (b_time_decays.squeeze()))

        attention = torch.tanh(torch.sigmoid(attention) / (denominator))


        attention = F.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        x = self.projection1(x)
        x = torch.tanh(x)
        x = self.projection2(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_dim, support=None, num_nodes=None):
        super(SpatialAttention, self).__init__()
        self.support = support
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.linear = LinearCustom()
        self.projection1 = nn.Linear(in_dim, in_dim)
        self.projection2 = nn.Linear(in_dim, in_dim)

    def forward(self, x, parameters):
        batch_size = x.shape[0]
        key = self.linear(x, parameters[0])
        value = self.linear(x, parameters[1])

        query = torch.cat(torch.split(x, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.head_size ** 0.5)

        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        x = self.projection1(x)
        x = F.relu(x)
        x = self.projection2(x)
        return x


class LinearCustom(nn.Module):

    def __init__(self):
        super(LinearCustom, self).__init__()

    def forward(self, inputs, parameters):
        weights, biases = parameters[0], parameters[1]

        return torch.matmul(inputs, weights) + biases.unsqueeze(2).repeat(1, 1,inputs.shape[2],  1)




class Model(nn.Module):
    def __init__(self, device, num_nodes, input_dim, output_dim, channels, dynamic, lag, horizon, supports,
                 memory_size,hidden_size,is_lr,proxies):
        super(Model, self).__init__()
        self.supports = supports
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.channels = channels
        self.dynamic = dynamic
        self.start_fc = nn.Linear(in_features=input_dim, out_features=self.channels)
        self.memory_size = memory_size
        self.hidden_size=hidden_size
        self.device=device
        self.is_lr=is_lr

        self.layers = nn.ModuleList(
            [
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes,
                      cuts=5,cut_size=4,cut_list_start=[0,2,4,6,8],  no_proxies=proxies,memory_size=memory_size,hidden_size=hidden_size,lag=lag),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes,
                      cuts=2, cut_size=3, cut_list_start=[0, 2], no_proxies=proxies, memory_size=memory_size,hidden_size=hidden_size,lag=lag),
                Layer(device=device, input_dim=channels, dynamic=dynamic, num_nodes=num_nodes,
                      cuts=1,cut_size=2, cut_list_start=[0], no_proxies=proxies, memory_size=memory_size,hidden_size=hidden_size,lag=lag),
            ])

        self.skip_layers = nn.ModuleList([
            nn.Linear(in_features=5 * channels, out_features=256),
            nn.Linear(in_features=2 * channels, out_features=256),
            nn.Linear(in_features=1 *channels, out_features=256),
        ])

        self.projections = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, horizon)])


        self.unit = dif_unit(hidden_size, num_nodes)

        self.projections_spa = nn.Sequential(*[
            nn.Linear(num_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes)])



    def hidden_init(self, batch_size):
        h0 = torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device)
        return h0


    def forward(self, x,adj):

        x_sq = x.squeeze()


        x_sq2=self.projections_spa(x_sq)
        x_spatio = torch.matmul(x_sq2, adj)
        x_trans=x_spatio.transpose(0,1)


        seq_len = x_trans.size(0)
        batch_size = x_trans.size(1)
        h_time, h_space = [], []
        hidden = self.hidden_init(batch_size)

        for t in range(seq_len):
            hidden, h_task = self.unit(x_trans[t, :, :], hidden)
            h_time.append(h_task[0])
            h_space.append(h_task[1])

        h_time = torch.stack(h_time, dim=0)
        h_space = torch.stack(h_space, dim=0)

        h_time = h_time.transpose(0, 1)
        h_space = h_space.transpose(0, 1)



        x = self.start_fc(x)

        batch_size = x.size(0)
        skip = 0

        for layer, skip_layer in zip(self.layers, self.skip_layers):
            x = layer(x, h_time, h_space)
            skip_inp = x.transpose(2, 1).reshape(batch_size, self.num_nodes,-1)
            skip = skip + skip_layer(skip_inp)

        x = torch.relu(skip)

        return self.projections(x).transpose(2, 1).unsqueeze(-1)


class Layer(nn.Module):
    def __init__(self, device, input_dim, num_nodes, cuts, cut_size,cut_list_start, dynamic, memory_size, no_proxies,hidden_size,lag):
        super(Layer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.cuts = cuts
        self.cut_size = cut_size
        self.cut_list_start=cut_list_start
        self.no_proxies = no_proxies
        self.proxies = nn.Parameter(torch.randn(1, cuts * no_proxies, self.num_nodes, input_dim).to(device),
                                    requires_grad=True).to(device)

        self.temporal_att = TemporalAttention(input_dim, num_nodes=num_nodes, cut_size=cut_size,device=device)
        self.spatial_att = SpatialAttention(input_dim, num_nodes=num_nodes)
        self.hidden_size=hidden_size
        self.lag=lag


        self.temporal_parameter_generators = nn.ModuleList([
            ParameterGenerator(hidden_size=hidden_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic,lag=lag,cut_size=cut_size,no_proxies=no_proxies) for _ in range(2)
        ])

        self.spatial_parameter_generators = nn.ModuleList([
            ParameterGenerator(hidden_size=hidden_size, input_dim=input_dim, output_dim=input_dim,
                               num_nodes=num_nodes, dynamic=dynamic,lag=lag,cut_size=cut_size,no_proxies=no_proxies) for _ in range(2)
        ])

        self.aggregator = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        ])

    def forward(self, x, h_time,h_space):
        batch_size = x.size(0)


        temporal_parameters = [layer(x, h_time) for layer in self.temporal_parameter_generators]
        spatial_parameters = [layer(x, h_space) for layer in self.spatial_parameter_generators]

        data_concat = []
        out = 0
        for i in range(self.cuts):
            t = x[:, self.cut_list_start[i] :self.cut_list_start[i] + self.cut_size, :, :]

            proxies = self.proxies[:, i * self.no_proxies: (i + 1) * self.no_proxies]
            proxies = proxies.repeat(batch_size, 1, 1, 1) + out
            t1 = torch.cat([proxies, t], dim=1)

            out = self.temporal_att(t1[:, :self.no_proxies, :, :], t1, t1, temporal_parameters)

            out = self.spatial_att(out, spatial_parameters)

            out = (self.aggregator(out) * out).sum(1, keepdim=True)
            data_concat.append(out)

        return torch.cat(data_concat, dim=1)

class ParameterGenerator(nn.Module):
    def __init__(self, hidden_size, input_dim, output_dim, num_nodes, dynamic,lag,cut_size,no_proxies):
        super(ParameterGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.lag=lag
        self.cut_size=cut_size
        self.no_proxies=no_proxies

        if self.dynamic:
            self.weight_generator = nn.Sequential(*[
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, input_dim * output_dim)
            ])
            self.bias_generator = nn.Sequential(*[
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
                nn.ReLU(),
                nn.Linear(5, output_dim)
            ])
            self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size, bias=True)
            self.T_transform = nn.Linear(lag, cut_size + no_proxies)
        else:
            self.weights = nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
            self.biases = nn.Parameter(torch.rand(input_dim), requires_grad=True)

    def forward(self, x, h_time):
        if self.dynamic:

            memory = torch.tanh(h_time)
            memory = memory.transpose(1, 2)
            memory = self.T_transform(memory)
            memory = torch.tanh(memory)
            memory = memory.transpose(1, 2)
            weights = self.weight_generator(memory).view(x.shape[0], self.cut_size + self.no_proxies, self.input_dim, self.output_dim)
            biases = self.bias_generator(memory).view(x.shape[0], self.cut_size +  self.no_proxies, self.output_dim)
        else:
            weights = self.weights
            biases = self.biases
        return weights, biases


class dif_unit(nn.Module):
    def __init__(self, hidden_size,num_nodes):
        super(dif_unit, self).__init__()

        self.num_nodes = num_nodes

        self.hidden_size = hidden_size

        self.hidden_transform = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

        self.input_transform = nn.Linear(num_nodes, 2 * hidden_size, bias=True)

        self.transform = nn.Linear(hidden_size * 2, hidden_size, bias=True)


        self.nowshare=nn.Linear(hidden_size,hidden_size,bias=True)

        self.stgateoutput=nn.Linear(2*hidden_size,hidden_size,bias=True)
        self.stgatemid=nn.Linear(hidden_size,hidden_size,bias=True)

        self.interactLinear=nn.Linear(hidden_size,hidden_size,bias=True)

    def forward(self, x, hidden):

        h_in = hidden

        now_input = self.input_transform(x)
        now_time,now_space = now_input[:, :].chunk(2, 1)

        last_h = self.hidden_transform(h_in)
        last_time,last_space=last_h[:,:].chunk(2,1)

        nowtime_interact = self.interactLinear(now_time * now_space)
        realnow_time = now_time - nowtime_interact
        realnow_space = now_space - nowtime_interact

        lasttime_interact = self.interactLinear(last_time * last_space)
        reallast_time = last_time - lasttime_interact
        reallast_space = last_space - lasttime_interact

        share = nowtime_interact * torch.tanh(self.nowshare(now_time+now_space))

        fi_time=realnow_time+torch.sigmoid(self.stgateoutput(torch.cat([realnow_time,reallast_time],dim=-1)))

        fi_space=realnow_space+torch.sigmoid(self.stgateoutput(torch.cat([realnow_space,reallast_space],dim=-1)))

        hmid=share+torch.tanh(self.stgatemid(fi_time+fi_space))

        h_time = fi_time
        h_space = fi_space

        return hmid, (h_time, h_space)

