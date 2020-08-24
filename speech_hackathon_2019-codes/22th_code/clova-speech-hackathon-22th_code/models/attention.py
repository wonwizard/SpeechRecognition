"""

Copyright 2017- IBM Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

        self.wc1 = nn.Linear(dim, dim//4)
        self.wc2 = nn.Linear(dim, dim//4)
        self.wc3 = nn.Linear(dim, dim//4)
        self.wc4 = nn.Linear(dim, dim//4)

        self.wo1 = nn.Linear(dim, dim//4)
        self.wo2 = nn.Linear(dim, dim//4)
        self.wo3 = nn.Linear(dim, dim//4)
        self.wo4 = nn.Linear(dim, dim//4)

        self.wk1 = nn.Linear(dim, dim//4)
        self.wk2 = nn.Linear(dim, dim//4)
        self.wk3 = nn.Linear(dim, dim//4)
        self.wk4 = nn.Linear(dim, dim//4)

        self.attn_combine = nn.Linear(dim, dim)

        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        
        c1 = self.wc1(context)
        c2 = self.wc2(context)
        c3 = self.wc3(context)
        c4 = self.wc4(context)
        
        o1 = self.wo1(output)
        o2 = self.wo2(output)
        o3 = self.wo3(output)
        o4 = self.wo4(output)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        #attn = torch.bmm(output, context.transpose(1, 2))
        att1 = torch.bmm(o1, c1.transpose(1, 2))
        att2 = torch.bmm(o2, c2.transpose(1, 2))
        att3 = torch.bmm(o3, c3.transpose(1, 2))
        att4 = torch.bmm(o4, c4.transpose(1, 2))

        if self.mask is not None:
            #attn.data.masked_fill_(self.mask, -float('inf'))
            att1.data.masked_fill_(self.mask, -float('inf'))
            att2.data.masked_fill_(self.mask, -float('inf'))
            att3.data.masked_fill_(self.mask, -float('inf'))
            att4.data.masked_fill_(self.mask, -float('inf'))
        
        #attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        att1 = F.softmax(att1.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        att2 = F.softmax(att2.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        att3 = F.softmax(att3.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        att4 = F.softmax(att4.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        attn = torch.cat((att1, att2, att3, att4), dim=2)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        #mix = torch.bmm(attn, context)
        mix1 = torch.bmm(att1, c1)
        mix2 = torch.bmm(att2, c2)
        mix3 = torch.bmm(att3, c3)
        mix4 = torch.bmm(att4, c4)

        concat = torch.cat((mix1, mix2, mix3, mix4), dim=2)
        mix = self.attn_combine(concat)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
