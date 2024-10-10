from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
import os
import math
import csv
from sklearn.metrics import f1_score
from tqdm import tqdm 


""" Contains the part of speech tagger class. """

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 

    As per the write-up, you may find it faster to use multiprocessing (code included). 

    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n // processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i: None for i in range(n)}
    probabilities = {i: None for i in range(n)}

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i + k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time() - start) / 60} minutes.")

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i + k], tags[i:i + k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time() - start) / 60} minutes.")

    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if
                     tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if
                         tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0] if eos_idxes else len(sent)
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx + 1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc / num_whole_sent if num_whole_sent > 0 else 0))
    print("Mean Probabilities: {}".format(sum(probabilities.values()) / n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))

    confusion_matrix(model.tag2idx, model.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc / num_whole_sent if num_whole_sent > 0 else 0, token_acc, sum(probabilities.values()) / n

def infer_sentences(model, sentences, idx_offset):
    predictions = {}
    for i, sentence in enumerate(sentences):
        pred_tags = model.inference(sentence)
        predictions[idx_offset + i] = pred_tags
    return predictions

def compute_prob(model, sentences, tags_list, idx_offset):
    probabilities = {}
    for i, (sentence, tags) in enumerate(zip(sentences, tags_list)):
        prob = model.sequence_probability(sentence, tags)
        probabilities[idx_offset + i] = prob
    return probabilities

class POSTagger():
    def __init__(self):
        self.unigram_counts = None
        self.bigram_counts = None
        self.trigram_counts = None
        self.quadgram_counts = None
        self.emission_counts = None
        self.all_tags = []
        self.tag2idx = {}
        self.idx2tag = {}
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.transition_probs = {}
        self.emission_probs = {}
        self.total_tags = 0
        self.total_tokens = 0
        self.glove_embeddings = {}

        self.k = 1e-5  # Add-k smoothing parameter

        # Linear interpolation weights (should sum to 1)
        self.l1 = 0.05  
        self.l2 = 0.15  
        self.l3 = 0.3   
        self.l4 = 0.5   
        self.suffix_tag_counts = {}
        self.suffix_total_counts = {}
        self.suffix_length = 3 

    def get_unigrams(self):
        """
        Computes unigram probabilities
        """
        num_tags = len(self.all_tags)

        # Add-k smoothing:
        self.unigram_probs_ml = (self.unigram_counts + self.k) / (self.total_tags + self.k * num_tags)

    def get_bigrams(self):
        """
        Computes bigram probabilities
        """
        num_tags = len(self.all_tags)
        unigram_counts = self.unigram_counts.reshape(-1, 1)

        # Add-k smoothing:
        self.bigram_probs_ml = (self.bigram_counts + self.k) / (unigram_counts + self.k * num_tags)

    def get_trigrams(self):
        """
        Computes trigram probabilities
        """
        num_tags = len(self.all_tags)
        bigram_counts = self.bigram_counts.reshape(num_tags, num_tags, 1)

        # Add-k smoothing:
        self.trigram_probs_ml = (self.trigram_counts + self.k) / (bigram_counts + self.k * num_tags)

    def get_quadgrams(self):
        """
        Computes quadgram probabilities 
        """
        num_tags = len(self.all_tags)
        trigram_counts = self.trigram_counts.reshape(num_tags, num_tags, num_tags, 1)

        # Add-k smoothing:
        self.quadgram_probs_ml = (self.quadgram_counts + self.k) / (trigram_counts + self.k * num_tags)


    def compute_transition_probs(self):
        """
        Computes the smoothed transition probabilities + linear interpolation
        """
        num_tags = len(self.all_tags)
        self.transition_probs = np.zeros((num_tags, num_tags, num_tags, num_tags))
        for i in range(num_tags):
            for j in range(num_tags):
                for k in range(num_tags):
                    for l in range(num_tags):
                        p1 = self.quadgram_probs_ml[i, j, k, l]
                        p2 = self.trigram_probs_ml[j, k, l]
                        p3 = self.bigram_probs_ml[k, l]
                        p4 = self.unigram_probs_ml[l]
                        self.transition_probs[i, j, k, l] = self.l1 * p1 + self.l2 * p2 + self.l3 * p3 + self.l4 * p4

    def get_emissions(self):
        """
        Computes emission probabilities + add-k smoothing
        """
        num_tags = len(self.all_tags)
        num_words = len(self.vocab)
        self.emission_probs = np.zeros((num_tags, num_words))
        for i in range(num_tags):
            tag_count = self.unigram_counts[i]
            self.emission_probs[i, :] = (self.emission_counts[i, :] + self.k) / (tag_count + self.k * num_words)


    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        """
        self.data = data
        sentences, tags = data

        self.vocab = set([w for s in sentences for w in s])
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.all_tags = list(set([t for tag_seq in tags for t in tag_seq]))
        self.tag2idx = {self.all_tags[i]: i for i in range(len(self.all_tags))}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Initialize:
        num_tags = len(self.all_tags)
        num_words = len(self.vocab)

        self.unigram_counts = np.zeros(num_tags)
        self.bigram_counts = np.zeros((num_tags, num_tags))
        self.trigram_counts = np.zeros((num_tags, num_tags, num_tags))
        self.quadgram_counts = np.zeros((num_tags, num_tags, num_tags, num_tags))
        self.emission_counts = np.zeros((num_tags, num_words))

        # Count:
        for sent, tag_seq in zip(sentences, tags):
            for i in range(len(tag_seq)):
                tag_idx = self.tag2idx[tag_seq[i]]
                word_idx = self.word2idx[sent[i]]

                self.unigram_counts[tag_idx] += 1

                self.emission_counts[tag_idx, word_idx] += 1

                # Bigram:
                if i > 0:
                    prev_tag_idx = self.tag2idx[tag_seq[i - 1]]
                    self.bigram_counts[prev_tag_idx, tag_idx] += 1

                # Trigram:
                if i > 1:
                    prev_prev_tag_idx = self.tag2idx[tag_seq[i - 2]]
                    prev_tag_idx = self.tag2idx[tag_seq[i - 1]]
                    self.trigram_counts[prev_prev_tag_idx, prev_tag_idx, tag_idx] += 1

                # Quadgram:
                if i > 2:
                    prev_prev_prev_tag_idx = self.tag2idx[tag_seq[i - 3]]
                    prev_prev_tag_idx = self.tag2idx[tag_seq[i - 2]]
                    prev_tag_idx = self.tag2idx[tag_seq[i - 1]]
                    self.quadgram_counts[prev_prev_prev_tag_idx, prev_prev_tag_idx, prev_tag_idx, tag_idx] += 1

        self.total_tags = np.sum(self.unigram_counts)
        self.total_tokens = np.sum(self.emission_counts)

        # Apply smoothing, compute probabilities
        self.get_unigrams()
        self.get_bigrams()
        self.get_trigrams()
        self.get_quadgrams()
        self.compute_transition_probs()
        self.get_emissions()

        self.suffix_tag_counts = {}
        self.suffix_total_counts = {}
        self.suffix_length = 3 # Adjust as needed

        # Count unigrams, bigrams, trigrams, emissions, and collect suffix statistics
        for sent, tag_seq in zip(sentences, tags):
            for i in range(len(tag_seq)):
                tag_idx = self.tag2idx[tag_seq[i]]
                word_idx = self.word2idx[sent[i]]

                # Collect suffix statistics
                word = sent[i]
                suffix = word[-self.suffix_length:] if len(word) >= self.suffix_length else word
                if suffix not in self.suffix_tag_counts:
                    self.suffix_tag_counts[suffix] = np.zeros(num_tags)
                    self.suffix_total_counts[suffix] = 0
                self.suffix_tag_counts[suffix][tag_idx] += 1
                self.suffix_total_counts[suffix] += 1

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence"""
        prob = 1.0
        num_tags = len(self.all_tags)
        for i in range(len(sequence)):
            word = sequence[i]
            word_idx = self.word2idx.get(word, None)
            tag = tags[i]
            tag_idx = self.tag2idx[tag]

            if word_idx is not None:
                emission_prob = self.emission_probs[tag_idx, word_idx]
            else:
                emission_probs = self.handle_unknown_word(word)
                emission_prob = emission_probs[tag_idx]

            if i > 2:
                prev_prev_prev_tag_idx = self.tag2idx[tags[i - 3]]
                prev_prev_tag_idx = self.tag2idx[tags[i - 2]]
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                transition_prob = self.transition_probs[prev_prev_prev_tag_idx, prev_prev_tag_idx, prev_tag_idx, tag_idx]
            elif i > 1:
                prev_prev_tag_idx = self.tag2idx[tags[i - 2]]
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                transition_prob = self.l2 * self.trigram_probs_ml[prev_prev_tag_idx, prev_tag_idx, tag_idx] + \
                                self.l3 * self.bigram_probs_ml[prev_tag_idx, tag_idx] + \
                                self.l4 * self.unigram_probs_ml[tag_idx]
            elif i > 0:
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                transition_prob = self.l3 * self.bigram_probs_ml[prev_tag_idx, tag_idx] + \
                                self.l4 * self.unigram_probs_ml[tag_idx]
            else:
                transition_prob = self.unigram_probs_ml[tag_idx]

            prob *= emission_prob * transition_prob

        return prob

    def handle_unknown_word(self, word):
        ''' Use suffix tree: '''
        num_tags = len(self.all_tags)
        suffix = word[-self.suffix_length:] if len(word) >= self.suffix_length else word

        if suffix in self.suffix_tag_counts:
            tag_counts = self.suffix_tag_counts[suffix]
            total_count = self.suffix_total_counts[suffix]
            emission_probs = (tag_counts + self.k) / (total_count + self.k * num_tags)
        else:
            emission_probs = np.ones(num_tags) / num_tags

        return emission_probs


    def inference(self, sequence, method='viterbi'):
        """Tags a sequence with part of speech tags."""
        if method == 'greedy':
            return self.beam_search_decode(sequence, beam_width=1)
        elif method == 'beam':
            return self.beam_search_decode(sequence)
        elif method == 'viterbi':
            return self.viterbi_trigram(sequence)
        else:
            raise ValueError('Unknown decoding method: {}'.format(method))


    def beam_search_decode(self, sequence, beam_width=3):
        """
        Beam search decoding implementation.
        """
        num_tags = len(self.all_tags)
        
        beam = [([], 0.0)]
        
        for i, word in enumerate(sequence):
            word_idx = self.word2idx.get(word, None)
            if word_idx is None:
                emission_probs = self.handle_unknown_word(word)
            else:
                emission_probs = self.emission_probs[:, word_idx]
            
            new_beam = []
            
            for path, log_prob in beam:
                prev_prev_prev_tag_idx = path[-3] if len(path) > 2 else None
                prev_prev_tag_idx = path[-2] if len(path) > 1 else None
                prev_tag_idx = path[-1] if len(path) > 0 else None
                
                for tag_idx in range(num_tags):
                    if prev_prev_prev_tag_idx is not None and prev_prev_tag_idx is not None and prev_tag_idx is not None:
                        # quadgram
                        trans_prob = self.transition_probs[prev_prev_prev_tag_idx, prev_prev_tag_idx, prev_tag_idx, tag_idx]
                    elif prev_prev_tag_idx is not None and prev_tag_idx is not None:
                        # trigram 
                        trans_prob = self.l2 * self.trigram_probs_ml[prev_prev_tag_idx, prev_tag_idx, tag_idx] + \
                                    self.l3 * self.bigram_probs_ml[prev_tag_idx, tag_idx] + \
                                    self.l4 * self.unigram_probs_ml[tag_idx]
                    elif prev_tag_idx is not None:
                        # bigram 
                        trans_prob = self.l3 * self.bigram_probs_ml[prev_tag_idx, tag_idx] + \
                                    self.l4 * self.unigram_probs_ml[tag_idx]
                    else:
                        # unigram 
                        trans_prob = self.unigram_probs_ml[tag_idx]
                    
                    emission_prob = emission_probs[tag_idx]
                    
                    # Compute total log probability to avoid float issues
                    total_log_prob = log_prob + math.log(trans_prob) + math.log(emission_prob)
                    
                    new_beam.append((path + [tag_idx], total_log_prob))
            
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
        
        # Select the best path from the beam (highest log probability)
        best_path, best_log_prob = max(beam, key=lambda x: x[1])
        
        tag_sequence = [self.idx2tag[tag_idx] for tag_idx in best_path]
        
        return tag_sequence

    def viterbi_decode(self, sequence):
        """
        Viterbi decoding implementation.
        """
        num_tags = len(self.all_tags)
        sequence_length = len(sequence)
        viterbi = np.zeros((sequence_length, num_tags))
        backpointer = np.zeros((sequence_length, num_tags), dtype=int)

        # Initialize DP:
        word_idx = self.word2idx.get(sequence[0], None)
        if word_idx is None:
            emission_probs = self.handle_unknown_word(sequence[0])
        else:
            emission_probs = self.emission_probs[:, word_idx]
        viterbi[0, :] = np.log(self.unigram_probs_ml) + np.log(emission_probs)

        for t in range(1, sequence_length):
            word_idx = self.word2idx.get(sequence[t], None)
            if word_idx is None:
                emission_probs = self.handle_unknown_word(sequence[t])
            else:
                emission_probs = self.emission_probs[:, word_idx]
            emission_log_probs = np.log(emission_probs)

            for s in range(num_tags):
                max_prob = None
                max_state = None
                for s_prev in range(num_tags):
                    transition_prob_log = np.log(self.bigram_probs_ml[s_prev, s])
                    prob = viterbi[t - 1, s_prev] + transition_prob_log + emission_log_probs[s]
                    if (max_prob is None) or (prob > max_prob):
                        max_prob = prob
                        max_state = s_prev
                viterbi[t, s] = max_prob
                backpointer[t, s] = max_state

        best_last_state = np.argmax(viterbi[sequence_length - 1, :])

        # Backtrack:
        best_path = [best_last_state]
        for t in range(sequence_length - 1, 0, -1):
            best_last_state = backpointer[t, best_last_state]
            best_path.insert(0, best_last_state)

        tag_sequence = [self.idx2tag[state] for state in best_path]

        return tag_sequence

    def viterbi_trigram(self, sequence):
        num_tags = len(self.all_tags)
        sequence_length = len(sequence)

        # Initialize π(0, *, *) to 1 for the start case and 0 for others
        pi = np.zeros((sequence_length + 1, num_tags, num_tags))  # π(k, u, v)
        bp = np.zeros((sequence_length + 1, num_tags, num_tags), dtype=int)  # Backpointers bp(k, u, v)
        
        pi[0, :, :] = 0  # Initialization to 0
        pi[0, self.tag2idx["*"], self.tag2idx["*"]] = 1  # π(0, *, *) = 1 for start of sentence

        # Loop over words in the sentence
        for k in range(1, sequence_length + 1):
            word = sequence[k - 1]
            word_idx = self.word2idx.get(word, None)
            if word_idx is None:
                emission_probs = self.handle_unknown_word(word)
            else:
                emission_probs = self.emission_probs[:, word_idx]

            for u in range(num_tags):  # Previous tag
                for v in range(num_tags):  # Current tag
                    max_prob = -float('inf')
                    best_w = None
                    for w in range(num_tags):  # Tag before u
                        # Transition probability q(v|w,u) and emission probability e(x_k|v)
                        trans_prob = self.trigram_probs_ml[w, u, v]
                        prob = pi[k - 1, w, u] * trans_prob * emission_probs[v]
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_w = w
                    
                    pi[k, u, v] = max_prob
                    bp[k, u, v] = best_w  # Store the best previous tag (w)
        
        # Backtracking to find the best tag sequence
        tags = [0] * sequence_length  # Placeholder for the predicted tags
        
        # Last tag pair (y_n-1, y_n) based on the argmax of π(n, u, v) × q(STOP|u, v)
        max_prob = -float('inf')
        best_u, best_v = None, None
        
        for u in range(num_tags):
            for v in range(num_tags):
                stop_prob = pi[sequence_length, u, v] * self.unigram_probs_ml[self.tag2idx['<STOP>']]
                if stop_prob > max_prob:
                    max_prob = stop_prob
                    best_u, best_v = u, v
        
        tags[-2] = best_u
        tags[-1] = best_v

        # Trace back the sequence from (n-2) to 1
        for k in range(sequence_length - 3, -1, -1):
            tags[k] = bp[k + 3, tags[k + 1], tags[k + 2]]
        
        # Convert indices to tags
        tag_sequence = [self.idx2tag[idx] for idx in tags]
        
        return tag_sequence

        


if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")
    dev_data_x = load_data("data/dev_x.csv")

    pos_tagger.train(train_data)

    # Evaluate on the development set
    evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        tags = pos_tagger.inference(sentence)
        test_predictions.extend(tags)

    # Write the predictions to a file
    results = []
    new_idx = 0
    for idx, tag in enumerate(test_predictions):
        # Ensure the tag is properly quoted
        if (str(tag) != "<STOP>"):
            results.append({'id': new_idx, 'tag': str(tag)})
            new_idx += 1


    # Create a DataFrame with the predictions
    df_predictions = pd.DataFrame(results)

    # Write them to a file to update the leaderboard
    df_predictions.to_csv('test_y.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
   
