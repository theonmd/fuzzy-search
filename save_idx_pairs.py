import time
import numpy as np
import json

filename = 'allfiles.txt'
with open(filename, 'rt') as ifile:
    corpus = ifile.readlines()
corpus = [line.rstrip() for line in corpus]
tokenized_corpus = [x.split() for x in corpus]

print('Start to index the words...')
word2idx = dict()
idx2word = dict()
w = 0
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in word2idx:
            word2idx[token] = w
            idx2word[str(w)] = token
            w += 1
            if w%1000 == 0:
                print(f'Got {w} words')
vocabulary_size = len(word2idx)
print(f'Vocabulary size: {vocabulary_size}')

tm = time.strftime('%Y%m%d_%H:%M:%S', time.localtime())

word2idx_fn = f'word2idx_{tm}.json'
with open(word2idx_fn, 'w') as word2idx_of:
    json.dump(word2idx, word2idx_of)

idx2word_fn = f'idx2word_{tm}.json'
with open(idx2word_fn, 'w') as idx2word_of:
    json.dump(idx2word, idx2word_of)

print('Finished saving word2idx and idx2word')


window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make sure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs)  # it will be useful to have this as numpy array
with open('idx_pairs.npy', 'wb') as f:
    np.save(f, idx_pairs)