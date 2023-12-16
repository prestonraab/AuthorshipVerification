import torch
from torch import nn
# from torchaudio.transforms import Spectrogram
#
# from transformers import DistilBertModel

from constants import *


class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        #
        # # self.cepstral = LFCC(n_lfcc=40)
        # self.pos_spectral = Spectrogram(n_fft=N_FFT)

        # Bert has hidden size 768
        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(768 * FFT_OUT, HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE, FINAL_VECTOR_SIZE)
        )
        # initialize the weights
        self.linear_gelu_stack.apply(self.init_weights)

    def forward_once(self, x):
        # # x starts as a dictionary of tensors.
        # # Each tensor is made of a number for each token in the input.
        #
        # input_ids, attention_mask = x['input_ids'], x['attention_mask']
        # x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        #
        # # Now x is a tensor of shape (batch_size, sequence_length, hidden_size)
        # # I want x to be (batch_size, hidden_size=768, sequence_length=512)
        # x = x.transpose(1, 2)
        # # cep
        # x = self.pos_spectral(x)
        # # Shape: (batch_size, hidden_size=768, c=fft_out=257, n_frames = 3)
        # x = x.mean(3)
        # # Shape: (batch_size, hidden_size=768, c=257)
        # x = x.flatten(1)
        # Shape: (batch_size, hidden_size=768 * 257 = 197376)
        logits = self.linear_gelu_stack(x)
        # Shape: (batch_size, final_vector_size=128)
        return logits

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)

        return output1, output2

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)