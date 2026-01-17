import torch
import torch.nn as nn


class SoftmaxGating(nn.Module):
    def __init__(self, input_emb_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_emb_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: int):
        return self.softmax(self.gate(x)) 


class NoisyTopKGating(nn.Module):
    def __init__(self, input_emb_dim: int, num_experts: int, k: int):
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.num_experts = num_experts
        self.k = k

        self.gate = nn.Linear(input_emb_dim, num_experts)
        self.noise = nn.Linear(input_emb_dim, num_experts)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        gate_output = self.gate(x)

        if self.training:
            noise_std = self.softplus(self.noise(x))
            noise_tensor = torch.randn_like(gate_output) * noise_std
            noise_logits = gate_output + noise_tensor
        else:
            noise_logits = gate_output # Deterministic behavior during inference

        top_k_logits, top_k_indices = torch.topk(noise_logits, self.k, dim=-1) # Gets topk logits and their indicies
        full_logits = torch.full_like(noise_logits, float('-inf')) # creates a tensor with same shape as input tensor with fill values
        full_logits.scatter_(-1, top_k_indices, top_k_logits) # replaces values in last dimension, in positions top_k_indices with values top_k_logits
        gate_probs = self.softmax(full_logits)

        return gate_probs, top_k_indices


class Expert(nn.Module):
    def __init__(self, input_dim: int, expert_hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(expert_hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, input_emb_dim: int, experts_output_dim: int, num_experts: int, k: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts_output_dim = experts_output_dim
        self.k = k

        self.experts = nn.ModuleList([
            Expert(input_emb_dim, experts_output_dim) 
            for _ in range(num_experts)
        ])
        self.gate = NoisyTopKGating(input_emb_dim, num_experts, k)

    def forward(self, x: torch.Tensor):
        gate_output, _ = self.gate(x)

        final_output = torch.zeros(x.size(0), self.experts_output_dim, device=x.device)

        for i, expert in enumerate(self.experts):
            expert_mask = gate_output[:, i] > 0
            if expert_mask.any():
                selection_indices = expert_mask.nonzero(as_tuple=True)[0] # Gets the indices of the experts
                selected_input = x[selection_indices] # Gets input to the selected experts 
                
                expert_output = expert(selected_input)
                gate_weight = gate_output[selection_indices, i].unsqueeze(1) # Gets output of gate
                
                weighted_output = expert_output * gate_weight # Weights expert output with gate output

                final_output.index_add_(0, selection_indices, weighted_output) # Accumulates the current expert's weighted output into the correct batch positions of the final tensor
                
        return final_output