import numpy as np
import time

def n_gram_train(label_file, n, example_nums=0):
    label_file = open(label_file)

    if example_nums == 0:
        dic_all = {}
        lines = label_file.readline()
        while lines:
            code = lines.split(',')[1]
            token = ['818'] + code.split(' ')[:-1] + ['819']
            target_key = ''
            if len(token) > n-1:  # sentence is long enough
                for i in range(n):
                    target_key = target_key + token[i] + ' '
                target_key = target_key[:-1]
                for i in range(len(token)-n):
                    if target_key not in dic_all:
                        dic_all[target_key] = 1
                    else:
                        dic_all[target_key] += 1
                    target_key = target_key[target_key.index(' ')+1:]+' ' + token[i+n]
                if target_key not in dic_all:
                    dic_all[target_key] = 1
                else:
                    dic_all[target_key] += 1
            lines = label_file.readline()
        label_file.close()
        return dic_all
    else:
        return n_gram_train_helper(label_file, n, example_nums)


def n_gram_train_helper(label_file, n, example_nums):
    dic_all = {}
    lines = label_file.readline()
    for trys in range(example_nums):
        code = lines.split(',')[1]
        token = ['818'] + code.split(' ')[:-1] + ['819']
        target_key = ''
        if len(token) > n-1:
            for i in range(n):
                target_key = target_key + token[i] + ' '
            target_key = target_key[:-1]
            for i in range(len(token)-n):
                if target_key not in dic_all:
                    dic_all[target_key] = 1
                else:
                    dic_all[target_key] +=1
                target_key = target_key[target_key.index(' ')+1:]+' ' + token[i+n]
            if target_key not in dic_all:
                dic_all[target_key] = 1
            else:
                dic_all[target_key] +=1
        lines = label_file.readline()
    label_file.close()
    return dic_all


def n_gram_infer(n_gram, qry):
    # infer p(xn | x1 ... xn-1)
    # qry: x1...xn-1 as a numpy array of size(1,n-1)
    # n_gram: dict
    # key: x1...xn as '111 222 333 444' / val: 1
    # output: p(xn | x1 ... xn-1) as a numpy array of size(w,1)
    # erase zero padding
    qry = qry[np.where(qry != 0)]

    # merge qry to qry_str
    qry_str = " ".join(str(e) for e in qry)
    qry_str += " "

    # count occurrence
    p = np.zeros(819)
    for i in range(819):
        cnt = n_gram.get(qry_str + str(i+1), 0)
        p[i] = cnt
        #if cnt != 0: print(qry_str + str(i+1) + " / " + str(cnt))

    # normalize
    if p.sum(0) != 0:
        p = p/p.sum(0)
    # smooth
    p[np.where(p == 0)] = 1e-6
    return p


def n_gram_p(n_gram_models, qry):
    # infer p(x_1 ... x_n)
    # n_gram_models: dict, key: 2 -> val: 2_gram
    # qry: a numpy array of size(1,n)
    # output p: sum(log p(x_1 ... x_i))

    # erase zero padding and add sos
    qry = qry[np.where(qry != 0)]
    n = qry.size
    qry = np.concatenate(([818], qry))
    print(qry)

    p_arr = np.zeros((n, 819))
    for i in range(n):
        if i+2 < 4:
            n_gram_size = i+2
            n_gram = n_gram_models[i+2]
        subseq = qry[i+2 - n_gram_size:i+1]  # x1...xn-1
        p_arr[i, :] = n_gram_infer(n_gram, subseq)  # p(xn | x1...xn-1)
    p_arr = p_arr[np.arange(p_arr.shape[0]), qry[1:]-1]

    logp = np.sum(np.log(p_arr)) / n
    return logp


'''
# examples
print(n_gram_train(label_file="train_label", n=2, example_nums=1))
print(n_gram_train(label_file="train_label", n=3, example_nums=1))
print(n_gram_train(label_file="train_label", n=5, example_nums=5))
print(n_gram_train(label_file="train_label", n=6))

print("Begin language model setup")
n_gram_models = {}
max_n_gram_size = 10
for n in range(max_n_gram_size-1):
    n_gram_models[n+2] = n_gram_train('train_label', n+2)
print("LM setup complete")

pred = 27
seq = [pred]
n = 2
while(pred != 819):
	n_gram_size = min(n, 5)
	subseq = seq[1-n_gram_size:]
	n_gram = n_gram_models[n_gram_size]
	p = n_gram_infer(n_gram, np.array(subseq))
	pred = np.argmax(p)+1
	seq.append(pred)
	n = n+1
print(seq)

start = time.time()
qry = np.array([27, 158, 130, 662, 621, 559, 15, 476, 662, 89, 480, 446, 662, 598, 620, 428, 661, 662, 819, 0, 0, 0, 0])
logp = n_gram_p(n_gram_models, qry)
print(logp)
end = time.time()
print(end - start)

qry = np.array([27, 158, 130, 662, 621, 559, 15, 476, 662, 89, 480, 446, 662, 598, 620, 428, 661, 662, 1, 0, 0, 0, 0])
logp = n_gram_p(n_gram_models, qry)
print(logp)

qry = np.array([27, 158, 130, 662, 621, 559, 15, 476, 662, 89, 480, 446, 662, 819, 1, 1, 1, 1, 1, 0, 0, 0, 0])
logp = n_gram_p(n_gram_models, qry)
print(logp)
'''

