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

import random

import numpy as np

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.n_gram import n_gram_p

class DecoderRNN(BaseRNN):
	r"""
	Provides functionality for decoding in a seq2seq framework, with an option for attention.
	Args:
		vocab_size (int): size of the vocabulary
		max_len (int): a maximum allowed length for the sequence to be processed
		hidden_size (int): the number of features in the hidden state `h`
		sos_id (int): index of the start of sentence symbol
		eos_id (int): index of the end of sentence symbol
		n_layers (int, optional): number of recurrent layers (default: 1)
		rnn_cell (str, optional): type of RNN cell (default: gru)
		bidirectional (bool, optional): if the encoder is bidirectional (default False)
		input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
		dropout_p (float, optional): dropout probability for the output sequence (default: 0)
		use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
	Attributes:
		KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
		KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
		KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`
	Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
		- **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
		  each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
		- **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
		  hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
		- **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
		  Used for attention mechanism (default is `None`).
		- **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
		  (default is `torch.nn.functional.log_softmax`).
		- **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
		  drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
		  teacher forcing would be used (default is 0).
	Outputs: decoder_outputs, decoder_hidden, ret_dict
		- **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
		  the outputs of the decoding function.
		- **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
		  state of the decoder.
		- **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
		  representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
		  predicted token IDs }.
	"""

	KEY_ATTN_SCORE = 'attention_score'
	KEY_LENGTH = 'length'
	KEY_SEQUENCE = 'sequence'

	def __init__(self, cfg_model, vocab_size, sos_id, eos_id):

		rnn_cell = cfg_model["rnn_cell"]
		max_len = cfg_model["dec"]["max_len"]
		hidden_size = cfg_model["hidden_size"] * (2 if cfg_model["bidirectional"] else 1)
		n_layers = cfg_model["dec"]["layer_size"]
		enc_n_layers = cfg_model["enc"]["layer_size"]
		bidirectional = cfg_model["bidirectional"]
		input_dropout_p = cfg_model["dropout"]
		dropout_p = cfg_model["dropout"]
		use_attention = cfg_model["dec"]["use_attention"]


		self.n_layers = n_layers

		if enc_n_layers < n_layers:
			# TODO: assume enc_n_layers > n_layers and slice?
			raise NotImplementedError("encoder must be at least as deep as decoder")

		super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
				input_dropout_p, dropout_p,
				n_layers, rnn_cell)

		self.bidirectional_encoder = bidirectional
		self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

		self.output_size = vocab_size
		self.max_length = max_len
		self.use_attention = use_attention
		self.eos_id = eos_id
		self.sos_id = sos_id

		self.init_input = None

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		if use_attention:
			self.attention = Attention(self.hidden_size)

		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward_step(self, input_var, hidden, encoder_outputs, function):
		batch_size = input_var.size(0)
		output_size = input_var.size(1)
		embedded = self.embedding(input_var)
		embedded = self.input_dropout(embedded)

		if self.training:
			self.rnn.flatten_parameters()

		output, hidden = self.rnn(embedded, hidden)

		attn = None
		if self.use_attention:
			output, attn = self.attention(output, encoder_outputs)

		predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
		return predicted_softmax, hidden, attn

	def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
					function=F.log_softmax, teacher_forcing_ratio=0, use_beam=False, ngram_models=None):
		
		ret_dict = dict()
		if self.use_attention:
			ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

		inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
															 function, teacher_forcing_ratio)

		decoder_hidden = self._init_state(encoder_hidden)

		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

		decoder_outputs = []
		sequence_symbols = []
		lengths = np.array([max_length] * batch_size)

		def decode(step, step_output, step_attn):
			decoder_outputs.append(step_output)
			if self.use_attention:
				ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
			symbols = decoder_outputs[-1].topk(1)[1]
			sequence_symbols.append(symbols)

			eos_batches = symbols.data.eq(self.eos_id)
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
			return symbols

		# Manual unrolling is used to support random teacher forcing.
		# If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
		if use_teacher_forcing:
			decoder_input = inputs[:, :-1]
			decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
																	 function=function)

			for di in range(decoder_output.size(1)):
				step_output = decoder_output[:, di, :]
				if attn is not None:
					step_attn = attn[:, di, :]
				else:
					step_attn = None
				decode(di, step_output, step_attn)
		elif not use_beam:
			# greedy decoding
			decoder_input = inputs[:, 0].unsqueeze(1)

			for di in range(max_length):
				decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
																		 function=function)
				step_output = decoder_output.squeeze(1)
				symbols = decode(di, step_output, step_attn)
				decoder_input = symbols
		else:
			# beam decoding
			SOS_idx = 818
			EOS_idx = 819

			beam_width = 4
			output_sequence = torch.zeros((batch_size, max_length), dtype=torch.int64)	
			for b in range(batch_size):
				# for each data in a batch, expand to beam_width dimension
				if(type(decoder_hidden)==tuple):
					b_decoder_hidden0 = decoder_hidden[0][:, b, :].unsqueeze(dim=1).expand((-1, beam_width, -1)).contiguous()
					b_decoder_hidden1 = decoder_hidden[1][:, b, :].unsqueeze(dim=1).expand((-1, beam_width, -1)).contiguous()
					b_decoder_hidden = (b_decoder_hidden0, b_decoder_hidden1)
				else:
					b_decoder_hidden = decoder_hidden[:, b, :].unsqueeze(dim=1).expand((-1, beam_width, -1)).contiguous()
				b_encoder_outputs = encoder_outputs[b, :, :].unsqueeze(dim=0).expand((beam_width, -1, -1)).contiguous()

				# implement beam decoding here
				hypothesis_beams = []
				hypothesis_logits = []

				# initialize beams
				b_decoder_input = torch.LongTensor([SOS_idx] * beam_width).view(beam_width, 1) # (BW, 1)
				b_decoder_input = b_decoder_input.to(device)
				reduced_beams = torch.zeros((beam_width, max_length), dtype=torch.int64, device=device) # (BW, L), no SOS
				reduced_logits = torch.zeros((beam_width, 1), device=device) # (BW, 1)

				for di in range(max_length):

					# obtain logits for each (beam, next token) pair
					b_decoder_output, b_decoder_hidden, b_step_attn = self.forward_step(
						b_decoder_input, b_decoder_hidden, b_encoder_outputs, function=function)
					b_step_output = b_decoder_output.squeeze(1) # (BW, voc_size)
					voc_size = b_step_output.size(1)

					# expand beams
					expanded_logits = reduced_logits.expand((-1, voc_size)).clone() # (BW, voc_size)
					expanded_logits += b_step_output
					expanded_beams = reduced_beams.unsqueeze(dim=1).expand((-1, voc_size, -1)).clone() # (BW, voc_size, L)
					all_next_tokens = torch.arange(voc_size, device=device).view((1, voc_size)).expand((beam_width, -1)) # (BW, voc_size)
					expanded_beams[:, :, di] = all_next_tokens

					# pop expanded beams ending with EOS and add to hypothesis
					hypothesis_beams.append(expanded_beams[:, EOS_idx, :])
					hypothesis_logits.append(expanded_logits[:, EOS_idx])
					expanded_beams = expanded_beams[:, :EOS_idx, :]
					expanded_logits = expanded_logits[:, :EOS_idx]

					# leave only best beams
					flat_logits = expanded_logits.contiguous().view((-1, 1)) # (BW * (voc_size-1), 1)
					flat_beams = expanded_beams.contiguous().view((beam_width*(voc_size-1), max_length)) # (BW * (voc_size-1), L)
					_, flat_idx = torch.topk(flat_logits, beam_width, dim=0)
					reduced_beams = flat_beams[flat_idx, :].squeeze(dim=1)
					reduced_logits = flat_logits[flat_idx, :].squeeze(dim=1)

					# prepare next iteration
					b_symbols = reduced_beams[:, di].unsqueeze(dim=1) # (BW, 1)
					b_decoder_input = b_symbols


				hyp_beams = torch.cat(hypothesis_beams[1:], dim=0) # (num_hyps, L) exclude '</s>'
				hyp_logits = torch.cat(hypothesis_logits[1:], dim=0) # (num_hyps)
				hyp_lengths = torch.arange(1, max_length, device=device, dtype=torch.float32).unsqueeze(dim=1).expand((-1, beam_width)).contiguous().view(-1) # (num_hyps)

				best_seq = rescoring(hyp_beams, hyp_logits, hyp_lengths, ngram_models) # (1, L)
				output_sequence[b, :] = best_seq

		ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
		ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

		output_sequence = output_sequence.cpu()

		return decoder_outputs, decoder_hidden, ret_dict, output_sequence

	def _init_state(self, encoder_hidden):
		""" Initialize the encoder hidden state. """
		if encoder_hidden is None:
			return None
		if isinstance(encoder_hidden, tuple):
			encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
		else:
			encoder_hidden = self._cat_directions(encoder_hidden)
		return encoder_hidden

	def _cat_directions(self, h):
		""" If the encoder is bidirectional, do the following transformation.
			(#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
		"""
		if self.bidirectional_encoder:
			h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
		if self.n_layers < h.size(0):
			h = h[:self.n_layers] # (n_layer, batch, dirs * hidden)
		return h

	def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
		if self.use_attention:
			if encoder_outputs is None:
				raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

		# inference batch size
		if inputs is None and encoder_hidden is None:
			batch_size = 1
		else:
			if inputs is not None:
				batch_size = inputs.size(0)
			else:
				if self.rnn_cell is nn.LSTM:
					batch_size = encoder_hidden[0].size(1)
				elif self.rnn_cell is nn.GRU:
					batch_size = encoder_hidden.size(1)

		# set default input and max decoding length
		if inputs is None:
			if teacher_forcing_ratio > 0:
				raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
			inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
			if torch.cuda.is_available():
				inputs = inputs.cuda()
			max_length = self.max_length
		else:
			max_length = inputs.size(1) - 1 # minus the start of sequence symbol

		return inputs, batch_size, max_length

def rescoring(hyp_beams, hyp_logits, hyp_lengths, ngram_models):
	# hyp_beams: (num_hyps, L)
	# hyp_logits: (num_hyps)
	# hyp_lengths: (num_hyps)
	num_hyps = list(hyp_beams.size())[0]
	score = hyp_logits / hyp_lengths
	ngram_w = 0.2

	use_ngram = (ngram_models is not None)
	if use_ngram:
		ngram_logits = np.zeros(num_hyps)
		for i in range(num_hyps):
			qry = hyp_beams[i, :].numpy().squeeze()
			ngram_logits[i] = n_gram_p(ngram_models, qry)
		score = score*(1-ngram_w) + torch.from_numpy(ngram_logits)*ngram_w
		print(ngram_logits)

	_, idx = torch.topk(score, 1)
	best_seq = hyp_beams[idx, :] # (1, L)

	return best_seq 
