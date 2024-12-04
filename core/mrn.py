import torch
import torchfile
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet34
from models.LSTM import BiLSTM_Attention
from torch.distributions.normal import Normal


def load_networks(networks, result_dir, name='', loss='', criterion=None):
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    checkpoint = torch.load(filename)
    new_state_dict = {}
    
    if(criterion == None):
        for k, v in checkpoint.items():
            new_state_dict[k[7:]] = v
    else:
        for k, v in checkpoint.items():
            new_state_dict[k] = v
            
    networks.load_state_dict( new_state_dict)
    
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        criterion.load_state_dict(torch.load(filename))
        
    return networks, criterion


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        if sum(self._part_sizes)!=inp_exp.shape[0]:
            temp = np.argmax(np.array(self._part_sizes))
            self._part_sizes[temp]+=(inp_exp.shape[0]-sum(self._part_sizes))
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
    
class MOEfusion(nn.Module):
    def __init__(self, num_classes, options,text_dim):
        super(MOEfusion, self).__init__()
        biL = BiLSTM_Attention(text_dim, 256, 2, options['num_classes'])
        res = ResNet34(options['num_classes'], options)
        self.net1 = res
        self.net2 = biL
        self.k = options['num_k']
        self.num_experts = options['num_exp']
        self.inputdim = 1024
        self.hidden_size = 2048
        self.mlpsize = options['num_classes']
        self.experts = nn.ModuleList([MLP(self.inputdim, self.mlpsize, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(self.inputdim, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.inputdim, self.num_experts), requires_grad=True)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.num_units =512
        self.num_heads = 8
        self.key_dim = 512
        self.mask_prob = 0.1
        self.cross_attn0=nn.MultiheadAttention(512, self.num_heads, batch_first=True)
        self.W_query = nn.Linear(in_features=512, out_features=self.num_units, bias=False)
        self.W_key = nn.Linear(in_features=512, out_features=self.num_units, bias=False)
        self.W_value = nn.Linear(in_features=512, out_features=self.num_units, bias=False)
        
        self.cross_attn1=nn.MultiheadAttention(512, self.num_heads,batch_first=True)
        self.W_query_1 = nn.Linear(in_features=512, out_features=self.num_units, bias=False)
        self.W_key_1 = nn.Linear(in_features=512, out_features=self.num_units, bias=False)
        self.W_value_1 = nn.Linear(in_features=512, out_features=self.num_units, bias=False)
        
        self.fc = nn.Linear(in_features=self.num_units, out_features=num_classes)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        if torch.isnan(clean_values).any():
            clean_values = torch.rand_like(clean_values)
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_idx = torch.arange(batch, device=clean_values.device) * m + self.k 
        threshold_v = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_idx), 1)
        is_in = torch.gt(noisy_values, threshold_v)
        threshold_positions_if_out = threshold_idx - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        if torch.isnan(clean_values).any():
            clean_values = torch.rand_like(clean_values)
        if torch.isnan(threshold_v).any():
            threshold_v = torch.rand_like(threshold_v)
        if torch.isnan(noise_stddev).any():
            noise_stddev = torch.rand_like(noise_stddev)
        if torch.isnan(threshold_if_out).any():
            threshold_if_out = torch.rand_like(threshold_if_out)
        prob_if_in = normal.cdf((clean_values - threshold_v) / noise_stddev) 
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-1):
        clean_logits = x @ self.w_gate
        if train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self,x1,x2,return_feature = False,loss_coef=1e-4):
        x1 = self.net1.model.conv1(x1)
        x1 = self.net1.model.bn1(x1)
        x1 = self.net1.model.relu(x1) 
        x1 = self.net1.model.maxpool(x1) 
        x1 = self.net1.model.layer1(x1)
        x1 = self.net1.model.layer2(x1)
        x1 = self.net1.model.layer3(x1) 
        x1 = self.net1.model.layer4(x1)
        feat1 = x1.flatten(-2).transpose(-2,-1)
        x2 = x2.type(torch.long)
        x2 = x2.view(len(x2), -1)
        x2 = x2.permute(1, 0)
        x2 = self.net2.embedding(x2)
        x2, _ = self.net2.encoder(x2)
        x2 = x2.permute(1, 0, 2)
        u = torch.tanh(torch.matmul(x2, self.net2.w_omega))
        att = torch.matmul(u, self.net2.u_omega)
        att_score = F.softmax(att, dim=1)
        x2 = x2 * att_score  
        feat2 = torch.sum(x2, dim=1) 
        
        feat2 = feat2[:,None]
        out = self.cross_attn0(
            query=feat2,
            key=feat1,
            value=feat1,
        )[0]   # [bs, 1, dim]
        
        out1=self.cross_attn1(
            query=feat1,
            key=feat2,
            value=feat2,
        )[0] 
        out1=torch.mean(out1, dim=1).squeeze()
        out=out.squeeze()

        out_cat= torch.concat((out,out1),1)
        
        gates, load = self.noisy_top_k_gating(out_cat, True)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(out_cat)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y= dispatcher.combine(expert_outputs)
        
        if return_feature:
            return out_cat, y, loss
        else:
            return y, loss