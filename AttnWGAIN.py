import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
def masked_mae_cal(inputs, target, mask):
    """ calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)

        v, attn_weights = self.attention(q, k, v, attn_mask)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, diagonal_attention_mask, device, dropout=0.1, attn_dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.device = device
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.seq_len).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        enc_output += residual
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Attention_WGAIN(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device,stage):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.stage = stage
        if self.stage == "G":
            actual_feature_num = feature_num * 3
        else :
            actual_feature_num = feature_num * 2

        self.layer_stack1 = nn.ModuleList([
            EncoderLayer(seq_len, actual_feature_num, d_model, d_inner, n_head, d_k, d_v, diagonal_attention_mask, device, dropout, dropout)
            for _ in range(n_groups)
        ])

        self.embedding_1 = nn.Linear(actual_feature_num, d_model)

        self.position_enc = PositionalEncoding(d_model, n_position=seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, feature_num)

    def forward(self, X,masks,delta=None):
        if self.stage == "G":
            x = torch.cat([X, delta], dim=2)
            input_X1 = torch.cat([x, masks], dim=2)
        else :
            input_X1 = torch.cat([X, masks], dim=2)
        input_X1 = self.embedding_1(input_X1)
        enc_output = self.dropout(self.position_enc(input_X1))
        for encoder_layer in self.layer_stack1:
            for _ in range(self.n_group_inner_layers):
                enc_output, _ = encoder_layer(enc_output)
        
        result_1 = self.reduce_dim(enc_output)
        result_temp = masks * X + (1 - masks) * result_1
        return result_temp


class Discriminator(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device):
        super(Discriminator, self).__init__()
        self.dim = feature_num
        self.h_dim = int(self.dim)
        self.D_W1 = nn.Parameter(init.xavier_normal_(torch.empty(self.dim*2, self.h_dim)))
        self.D_b1 = nn.Parameter(torch.zeros(self.h_dim))
        self.D_W2 = nn.Parameter(init.xavier_normal_(torch.empty([self.h_dim, self.h_dim])))
        self.D_b2 = nn.Parameter(torch.zeros(self.h_dim))
        self.D_W3 = nn.Parameter(init.xavier_normal_(torch.empty([self.h_dim, self.dim])))
        self.D_b3 = nn.Parameter(torch.zeros(self.dim))
        
    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=2)
        D_h1 = F.relu(torch.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = torch.matmul(D_h2, self.D_W3) + self.D_b3  
        D_prob = nn.Sigmoid()(D_logit)
        return D_prob

class Generator(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device):
        super(Generator, self).__init__()
        self.encoder=Attention_WGAIN(n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device,stage="G")
        self.encoder2=Attention_WGAIN(n_groups, n_group_inner_layers, seq_len, feature_num, d_model, d_inner, n_head, d_k, d_v, dropout, diagonal_attention_mask, device,stage="D")


    def forward(self, x, m, delta = None):
        if delta is None:
            output = self.encoder2(x, m)
        else:
            output = self.encoder(x, m, delta)
        return output


class D_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, M, G_sample, D_prob, X_holdout, indicating_mask, alpha):
        return torch.mean((1-M) * D_prob) - torch.mean(M * D_prob)


class G_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, M, G_sample, D_prob, X_holdout, indicating_mask, alpha):
        d_loss_real = -torch.mean((1 - M) * D_prob)
        Construction_MSE_loss = torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)
        impution_MSE_loss = torch.mean((indicating_mask * X_holdout - indicating_mask * G_sample) ** 2) / torch.mean(indicating_mask)
        return d_loss_real + alpha[0] * Construction_MSE_loss + alpha[1] * impution_MSE_loss
