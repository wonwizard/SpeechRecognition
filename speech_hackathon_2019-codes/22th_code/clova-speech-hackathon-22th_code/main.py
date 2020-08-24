"""
Copyright 2019-present NAVER Corp.

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

#-*- coding: utf-8 -*-

import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev
from pympler.tracker import SummaryTracker
from torch.optim.lr_scheduler import StepLR
tracker = SummaryTracker()

import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq
from models.n_gram import n_gram_train, n_gram_infer

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET

import config.utils

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

if HAS_DATASET == False:
	DATASET_PATH = './train'
	#DATASET_PATH = './sample_dataset'
	#DATASET_PATH = '../data'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')



def label_to_string(labels):
	if len(labels.shape) == 1:
		sent = str()
		for i in labels:
			if i.item() == EOS_token:
				break
			sent += index2char[i.item()]
		return sent

	elif len(labels.shape) == 2:
		sents = list()
		for i in labels:
			sent = str()
			for j in i:
				if j.item() == EOS_token:
					break
				sent += index2char[j.item()]
			sents.append(sent)

		return sents


def char_distance(ref, hyp):
	ref = ref.replace(' ', '') 
	hyp = hyp.replace(' ', '') 

	dist = Lev.distance(hyp, ref)
	length = len(ref.replace(' ', ''))

	return dist, length 


def get_distance(ref_labels, hyp_labels, display=False):
	total_dist = 0
	total_length = 0
	for i in range(len(ref_labels)):
		ref = label_to_string(ref_labels[i])
		hyp = label_to_string(hyp_labels[i])
		dist, length = char_distance(ref, hyp)
		total_dist += dist
		total_length += length 
		if display:
			cer = total_dist / total_length
			logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
	return total_dist, total_length


def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1.0):
	total_loss = 0.
	total_num = 0
	total_dist = 0
	total_length = 0
	total_sent_num = 0
	batch = 0

	model.train()

	logger.info('train() start')

	begin = epoch_begin = time.time()

	while True:
		if queue.empty():
			logger.debug('queue is empty')

		feats, scripts, feat_lengths, script_lengths = queue.get()

		if feats.shape[0] == 0:
			# empty feats means closing one loader
			train_loader_count -= 1

			logger.debug('left train_loader: %d' % (train_loader_count))

			if train_loader_count == 0:
				break
			else:
				continue

		optimizer.zero_grad()

		feats = feats.to(device)
		scripts = scripts.to(device)

		src_len = scripts.size(1)
		target = scripts[:, 1:]

		model.module.flatten_parameters()
		logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

		logit = torch.stack(logit, dim=1).to(device)

		y_hat = logit.max(-1)[1]

		loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
		total_loss += loss.item()
		total_num += sum(feat_lengths)

		display = random.randrange(0, 100) == 0
		dist, length = get_distance(target, y_hat, display=display)
		total_dist += dist
		total_length += length

		total_sent_num += target.size(0)

		loss.backward()
		optimizer.step()

		if batch % print_batch == 0:
			current = time.time()
			elapsed = current - begin
			epoch_elapsed = (current - epoch_begin) / 60.0
			train_elapsed = (current - train_begin) / 3600.0

			logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
				.format(batch,
						#len(dataloader),
						total_batch_size,
						total_loss / total_num,
						total_dist / total_length,
						elapsed, epoch_elapsed, train_elapsed))
			begin = time.time()

			nsml.report(False,
						step=train.cumulative_batch_count, train_step__loss=total_loss/total_num,
						train_step__cer=total_dist/total_length)
		batch += 1
		train.cumulative_batch_count += 1


	logger.info('train() completed')
	return total_loss / total_num, total_dist / total_length


train.cumulative_batch_count = 0


def evaluate(model, dataloader, queue, criterion, device, ngram_models=None):
	logger.info('evaluate() start')
	total_loss = 0.
	total_num = 0
	total_dist = 0
	total_length = 0
	total_sent_num = 0

	model.eval()

	with torch.no_grad():
		while True:
			feats, scripts, feat_lengths, script_lengths = queue.get()
			if feats.shape[0] == 0:
				break

			feats = feats.to(device)
			scripts = scripts.to(device)

			src_len = scripts.size(1)
			target = scripts[:, 1:]

			model.module.flatten_parameters()

			USE_BEAM = True

			if not USE_BEAM:
				logit, _ = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0, use_beam=USE_BEAM, ngram_models=ngram_models)
				logit = torch.stack(logit, dim=1).to(device)
				y_hat = logit.max(-1)[1]

				loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
				total_loss += loss.item()
				total_num += sum(feat_lengths)
			else:
				_, out_seq = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0, use_beam=USE_BEAM, ngram_models=ngram_models)
				y_hat = out_seq
				total_num = 1

			display = random.randrange(0, 100) == 0
			dist, length = get_distance(target, y_hat, display=display)
			total_dist += dist
			total_length += length
			total_sent_num += target.size(0)

	logger.info('evaluate() completed')
	return total_loss / total_num, total_dist / total_length


def bind_model(cfg_data, model, optimizer=None, ngram_models=None):
	def load(filename, **kwargs):
		state = torch.load(os.path.join(filename, 'model.pt'))
		model.load_state_dict(state['model'])
		if 'optimizer' in state and optimizer:
			optimizer.load_state_dict(state['optimizer'])
		print('Model loaded')

	def save(filename, **kwargs):
		state = {
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict()
		}
		torch.save(state, os.path.join(filename, 'model.pt'))

	def infer(wav_path):
		model.eval()
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		input = get_spectrogram_feature(cfg_data, wav_path).unsqueeze(0)
		input = input.to(device)

		USE_BEAM = True
		if not USE_BEAM:
			logit, _ = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0, use_beam=False, ngram_models=ngram_models)
			logit = torch.stack(logit, dim=1).to(device)
			y_hat = logit.max(-1)[1]
		else:
			_, out_seq = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0, use_beam=True, ngram_models=ngram_models)
			y_hat = out_seq

		hyp = label_to_string(y_hat)

		return hyp[0]

	nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.


def split_dataset(cfg, wav_paths, script_paths, valid_ratio=0.05):
	train_loader_count = cfg["workers"]
	records_num = len(wav_paths)
	batch_num = math.ceil(records_num / cfg["batch_size"])

	valid_batch_num = math.ceil(batch_num * valid_ratio)
	train_batch_num = batch_num - valid_batch_num

	batch_num_per_train_loader = math.ceil(train_batch_num / cfg["workers"])

	train_begin = 0
	train_end_raw_id = 0
	train_dataset_list = list()

	for i in range(cfg["workers"]):

		train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

		train_begin_raw_id = train_begin * cfg["batch_size"]
		train_end_raw_id = train_end * cfg["batch_size"]

		train_dataset_list.append(BaseDataset(
										cfg["data"],
										wav_paths[train_begin_raw_id:train_end_raw_id],
										script_paths[train_begin_raw_id:train_end_raw_id],
										SOS_token, EOS_token, train_mode=True))

		train_begin = train_end 

	valid_dataset = BaseDataset(cfg["data"], wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token, train_mode=False)

	return train_batch_num, train_dataset_list, valid_dataset


def main():

	global char2index
	global index2char
	global SOS_token
	global EOS_token
	global PAD_token

	parser = argparse.ArgumentParser(description='Speech hackathon Baseline')

	parser.add_argument('--no_train', action='store_true', default=False)
	parser.add_argument('--local', action='store_true', default=False)
	parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument("--pause", type=int, default=0)
	parser.add_argument("--USE_LM", action='store_true', default=False)
	parser.add_argument('--config', type=str, default='./config/legacy/cfg0/baseline.cfg0.json')
	args = parser.parse_args()
	cfg = config.utils.read_cfg(args.config)

	char2index, index2char = label_loader.load_label('./hackathon.labels')
	SOS_token = char2index['<s>']
	EOS_token = char2index['</s>']
	PAD_token = char2index['_']

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	args.cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device('cuda' if args.cuda else 'cpu')
    
	ngram_models = None

	if args.USE_LM:
		print("Begin language model setup")
		ngram_models = {}        
		max_n_gram_size = 4
		for n in range(max_n_gram_size-1):
			ngram_models[n+2] = n_gram_train(os.path.join(DATASET_PATH, 'train_label'), n+2)
			del(n)
		print("LM setup complete")

    
	# N_FFT: defined in loader.py
	feature_size = N_FFT / 2 + 1

	enc = EncoderRNN(
			cfg["model"],
			feature_size,
			variable_lengths=False)

	dec = DecoderRNN(
			cfg["model"],
			len(char2index),
			SOS_token, EOS_token)

	model = Seq2seq(enc, dec)
	model.flatten_parameters()

	for param in model.parameters():
		param.data.uniform_(-0.08, 0.08)

	model = nn.DataParallel(model).to(device)

	optimizer = optim.Adam(model.module.parameters(), lr=cfg["lr"])
	criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

	bind_model(cfg["data"], model, optimizer, ngram_models)
	if args.no_train and not args.local:
		nsml.load(checkpoint='best',session="team161/sr-hack-2019-50000/78")

	if args.pause == 1:
		nsml.paused(scope=locals())

	if args.mode != "train":
		return

	data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
	wav_paths = list()
	script_paths = list()

	with open(data_list, 'r') as f:
		for line in f:
			# line: "aaa.wav,aaa.label"

			wav_path, script_path = line.strip().split(',')
			wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
			script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

	best_loss = 1e10
	best_cer = 1e10
	begin_epoch = 0

	# load all target scripts for reducing disk i/o
	target_path = os.path.join(DATASET_PATH, 'train_label')
	load_targets(target_path)

	if args.no_train:
		train_batch_num, train_dataset_list, valid_dataset = split_dataset(cfg, wav_paths, script_paths,
		valid_ratio=0.05)
	else:
		train_batch_num, train_dataset_list, valid_dataset = split_dataset(cfg, wav_paths, script_paths, valid_ratio=0.05)

	lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.96)

	logger.info('start')

	nsml.save('notrain')

	train_begin = time.time()
	for epoch in range(begin_epoch, cfg["max_epochs"]):
		print("epoch", epoch)
		#tracker.print_diff()
		if not args.no_train:
			train_queue = queue.Queue(cfg["workers"] * 2)
			train_loader = MultiLoader(train_dataset_list, train_queue, cfg["batch_size"], cfg["workers"])
			train_loader.start()
			# scheduled sampling
			# ratio_s -> ratio_e (linear decreasing) -> maintain
			# decreasing epoch-scale = n_epoch_ramp
			n_epoch_ramp = 10
			ratio_s = 0.25
			ratio_e = 0
			teacher_forcing_ratio = max(ratio_s - (ratio_s-ratio_e)*epoch/n_epoch_ramp, ratio_e)
			train_loss, train_cer = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, cfg["workers"], 10, teacher_forcing_ratio)  # cfg["teacher_forcing"]
			lr_scheduler.step(epoch)
			logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
			train_loader.join()

		valid_queue = queue.Queue(cfg["workers"] * 2)
		valid_loader = BaseDataLoader(valid_dataset, valid_queue, cfg["batch_size"], 0)
		valid_loader.start()
		print("start eval")
		eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device, ngram_models=ngram_models)
		logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))
		valid_loader.join()
		print("end eval")

		if args.no_train:
			continue

		nsml.report(False,
					step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer,
					eval__loss=eval_loss, eval__cer=eval_cer)

		# save every epoch
		save_name = "model_%03d"%(epoch)
		nsml.save(save_name)
		# save best loss model
		is_best_loss = (eval_loss < best_loss)
		if is_best_loss:
			nsml.save('best')
			best_loss = eval_loss
		# save best cer model
		is_best_cer = (eval_cer < best_cer)
		if is_best_cer:
			nsml.save('cer')
			best_cer = eval_cer



if __name__ == "__main__":
	main()

