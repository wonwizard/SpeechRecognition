import os
import threading
import torch
import math
import random
from nsml.constants import DATASET_PATH

from utils.define import logger


class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    # 큐에 들어가는 batch를 만드는 함수
    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)  #  3차원의 0 벡터
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break
                #logger.info('BaseDataLoader 들어옴')
                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                # 큐에 batch 삽입
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            # 큐에 batch 삽입
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

# BaseLoader()를 여러 개 호출하는 클래스
class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        # BaseDataLoader run()을 실행!!
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()


def feed_infer(output_file, infer_func):

    filepath = os.path.join(DATASET_PATH, 'test', 'test_data', 'test_list.csv')

    with open(output_file, 'w') as of:

        with open(filepath, 'r') as f:

            for no, line in enumerate(f):

                # line : "abc.wav"

                wav_path = line.strip()
                wav_path = os.path.join(DATASET_PATH, 'test', 'test_data', wav_path)
                pred = infer_func(wav_path)

                of.write('%s,%s\n' % (wav_path, pred))
                print(wav_path, pred)

def _collate_fn(batch):
    PAD = 0
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    return seqs, targets, seq_lengths, target_lengths