import Levenshtein
import difflib
from thefuzz import fuzz
from thefuzz import process
import io
import re
import sys
import string
import tqdm
import json
import torch
import numpy as np
import itertools
from bs4 import BeautifulSoup

# the threshold for similarity score is set to be 0.99
SIM_THRESHOLD = 0.99

with open('word2idx_20220417_20:09:42.json') as d1:
    word2idx = json.load(d1)
with open('idx2word_20220417_20:09:42.json') as d2:
    idx2word = json.load(d2)

# import the word2vec model with the best performance (lowest loss value)
W2= torch.load('W2_44.pt')


# reads the input file and parse into sentences.
def clean_txt(filepath):
    whole_text = filepath.read()
    whole_text = whole_text.lower()
    soup = BeautifulSoup(whole_text, "html.parser")
    lines = [x.get_text() for x in soup.find_all("p")]
    lines = [l.replace(u'\xa0', u' ').replace('\n', ' ') for l in lines]
    lines = [l for l in lines if len(l.split()) > 0]
    sentences = []
    for line in lines:
        sentences += line.split('. ')
    return sentences

# measure the cosine similarity between two words
def cos_similarity(v,u):
    return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))

# returns a list of similar words and their similarity scores
def get_similar_words(word):
    #print(f'Getting similar words for \'{word}\'...')
    word_queue = {word: 1.0}
    if word not in word2idx: # if a word is not in the dictionary, the only word in the queue is itself
        return word_queue   
    word_vector = W2[word2idx[word]]
    for w in idx2word:
        if w == word2idx[word]:
            continue
        sim_score = cos_similarity(word_vector, W2[int(w)]).item()
        if sim_score >= SIM_THRESHOLD:
            word_queue[idx2word[w]] = sim_score
    #print('Finished!')
    return word_queue

# performs fuzzy search of the input query on the provided sentences list
def fuzzy_search(query, doc):
    #print('Executing fuzzy_search...')
    q_words = query.split()
    num_words = len(q_words)
    word_queues = []
    for word in q_words:
        word_queues.append(get_similar_words(word))
    query_list = list(itertools.product(*word_queues)) # generate the cross product of similar words lists
    query_queue = dict()
    for q in query_list:
        score = 0
        for i in range(num_words):
            score += word_queues[i][q[i]]
        score = score / num_words
        query_queue[' '.join(q)] = score
    query_queue = dict(sorted(query_queue.items(), key=lambda x: x[1], reverse=True))
    #print('Query queue:')
    #print(list(query_queue.items())[:5])

    final_results_raw = []
    for q in query_queue.keys():
        #print('extracting results for \'' + q + '\'')
        result_raw = process.extract(q, doc, scorer=fuzz.token_set_ratio)
        #print('Finished extracting')
        for r in result_raw:
            weighted_score = r[1] * (query_queue[q]*50-49) #transform the .99-1 range to .5-1 range
            if weighted_score > 100:
                weighted_score = 100
            isNewResult = True
            for fr in final_results_raw:
                if r[0] == fr[1]:
                    isNewResult = False
                    if weighted_score > fr[2]:
                        final_results_raw.remove(fr)
                        final_results_raw.append((q, r[0], weighted_score))
                    break
            if isNewResult:
                final_results_raw.append((q, r[0], weighted_score))

    final_results_sorted = sorted(final_results_raw, key=lambda x: x[2], reverse=True)
    return final_results_sorted

def main():
    filename = input('Copy path of the financial file here (txt format): ')
    try:
        f = open(filename, 'r')
    except IOError:
        sys.exit('File does not exist!')
    print('Processing file...')
    doc = clean_txt(f)
    f.close()

    while True:
        print('-----------------------------------------------------------')
        query = input('What are you searching for? ')
        print('Searching...')
        query = query.lower()
        sorted_results = fuzzy_search(query, doc)

        curr_display = 0
        while True:
            print('-----------------------------------------------------------')
            for r in sorted_results[curr_display:min(curr_display+5, len(sorted_results))]:
                print('Adjusted keywords: ' + r[0])
                print('Search result:')
                print(r[1])
                print('Matching score: ' + str(round(r[2], 4)) + '\n')

            curr_display += 5
            if curr_display >= len(sorted_results):
                print('Reached end of result list!')
                break
            
            cont_ans = input('Show more results? (Y/N): ')
            while cont_ans.lower() not in ('y', 'n'):
                cont_ans = input('Please enter Y or N: ')
            if cont_ans.lower() == 'n':
                break

        ans = input('Anything else to search? (Y/N): ')
        while ans.lower() not in ('y', 'n'):
            ans = input('Please enter Y or N: ')
        if ans.lower() == 'n':
            break
    
    print('Thank you for using!')


main()
