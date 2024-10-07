from multiprocessing import Pool
import numpy as np
import time
from tagger_utils import *
import os
import math
import csv

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
        """Initializes the tagger model parameters and anything else necessary. """
        self.unigram_counts = None
        self.bigram_counts = None
        self.trigram_counts = None
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
        # Smoothing parameters
        self.k = 1e-6  # Add-k smoothing parameter
        # Linear interpolation weights (should sum to 1)
        self.l1 = 0.1
        self.l2 = 0.3
        self.l3 = 0.6

    def get_unigrams(self):
        """
        Computes unigram probabilities with Add-k smoothing applied directly onto unigram_probs_ml.
        """
        num_tags = len(self.all_tags)
        # Add-k smoothing applied directly
        self.unigram_probs_ml = (self.unigram_counts + self.k) / (self.total_tags + self.k * num_tags)

    def get_bigrams(self):
        """
        Computes bigram probabilities with Add-k smoothing applied directly onto bigram_probs_ml.
        """
        num_tags = len(self.all_tags)
        # Reshape unigram counts to match bigram_counts dimensions
        unigram_counts = self.unigram_counts.reshape(-1, 1)
        # Add-k smoothing applied directly
        self.bigram_probs_ml = (self.bigram_counts + self.k) / (unigram_counts + self.k * num_tags)

    def get_trigrams(self):
        """
        Computes trigram probabilities with Add-k smoothing applied directly onto trigram_probs_ml.
        """
        num_tags = len(self.all_tags)
        # Reshape bigram counts to match trigram_counts dimensions
        bigram_counts = self.bigram_counts.reshape(num_tags, num_tags, 1)
        # Add-k smoothing applied directly
        self.trigram_probs_ml = (self.trigram_counts + self.k) / (bigram_counts + self.k * num_tags)

    def compute_transition_probs(self):
        """
        Computes the smoothed transition probabilities using linear interpolation.
        """
        num_tags = len(self.all_tags)
        self.transition_probs = np.zeros((num_tags, num_tags, num_tags))
        for i in range(num_tags):
            for j in range(num_tags):
                for k in range(num_tags):
                    p1 = self.trigram_probs_ml[i, j, k]
                    p2 = self.bigram_probs_ml[j, k]
                    p3 = self.unigram_probs_ml[k]
                    self.transition_probs[i, j, k] = self.l1 * p1 + self.l2 * p2 + self.l3 * p3

    def get_emissions(self):
        """
        Computes emission probabilities with Add-k smoothing.
        """
        num_tags = len(self.all_tags)
        num_words = len(self.vocab)
        self.emission_probs = np.zeros((num_tags, num_words))
        for i in range(num_tags):
            tag_count = self.unigram_counts[i]
            self.emission_probs[i, :] = (self.emission_counts[i, :] + self.k) / (tag_count + self.k * num_words)

    def load_glove_embeddings(self, glove_file='glove.6B.50d.txt'):
        """
        Load GloVe embeddings.
        """
        print("Loading GloVe embeddings for unknown words...")
        embedding_dim = 50 
        self.embedding_dim = embedding_dim
        self.glove_embeddings = {}

        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.glove_embeddings[word] = vector

        print("GloVe embeddings loaded.")

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.

        """
        self.data = data
        sentences, tags = data

        # Build the vocabulary
        self.vocab = set([w for s in sentences for w in s])
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        # Build tag mappings
        self.all_tags = list(set([t for tag_seq in tags for t in tag_seq]))
        self.tag2idx = {self.all_tags[i]: i for i in range(len(self.all_tags))}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Initialize counts
        num_tags = len(self.all_tags)
        num_words = len(self.vocab)

        self.unigram_counts = np.zeros(num_tags)
        self.bigram_counts = np.zeros((num_tags, num_tags))
        self.trigram_counts = np.zeros((num_tags, num_tags, num_tags))
        self.emission_counts = np.zeros((num_tags, num_words))

        # Count unigrams, bigrams, trigrams, and emissions
        for sent, tag_seq in zip(sentences, tags):
            for i in range(len(tag_seq)):
                tag_idx = self.tag2idx[tag_seq[i]]
                word_idx = self.word2idx[sent[i]]

                # Unigram:
                self.unigram_counts[tag_idx] += 1

                # Emission:
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

        self.total_tags = np.sum(self.unigram_counts)
        self.total_tokens = np.sum(self.emission_counts)

        # Apply smoothing and compute probabilities
        self.get_unigrams()
        self.get_bigrams()
        self.get_trigrams()
        self.compute_transition_probs()
        self.get_emissions()

        # Load GloVe embeddings for unknown words only
        glove_path = 'glove.6B.50d.txt'
        if os.path.exists(glove_path):
            self.load_glove_embeddings(glove_file=glove_path)
        else:
            print("GloVe embeddings not found. Please ensure the GloVe file is in the current directory.")

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition probabilities."""
        prob = 1.0
        num_tags = len(self.all_tags)
        for i in range(len(sequence)):
            word = sequence[i]
            word_idx = self.word2idx.get(word, None)
            tag = tags[i]
            tag_idx = self.tag2idx[tag]

            # Emission probability
            if word_idx is not None:
                emission_prob = self.emission_probs[tag_idx, word_idx]
            else:
                emission_probs = self.handle_unknown_word(word)
                emission_prob = emission_probs[tag_idx]

            # Transition probability
            if i > 1:
                prev_prev_tag_idx = self.tag2idx[tags[i - 2]]
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                transition_prob = self.transition_probs[prev_prev_tag_idx, prev_tag_idx, tag_idx]
            elif i > 0:
                prev_tag_idx = self.tag2idx[tags[i - 1]]
                transition_prob = self.l2 * self.bigram_probs_ml[prev_tag_idx, tag_idx] + \
                                  self.l3 * self.unigram_probs_ml[tag_idx]
            else:
                transition_prob = self.unigram_probs_ml[tag_idx]

            prob *= emission_prob * transition_prob

        return prob

    def handle_unknown_word(self, word):
        """
        Handle unknown words by estimating emission probabilities using GloVe embeddings.
        """
        num_tags = len(self.all_tags)
        # If GloVe embeddings are available
        if word in self.glove_embeddings:
            word_embedding = self.glove_embeddings[word]
        else:
            # For words not in GloVe, use a zero vector
            word_embedding = np.zeros(self.embedding_dim)

        # Estimate emission probabilities based on similarity with tag embeddings
        # For simplicity, let's assume each tag has an average embedding of words associated with it
        # We'll precompute this if not already done
        if not hasattr(self, 'tag_embeddings'):
            self.compute_tag_embeddings()

        emission_scores = np.zeros(num_tags)
        for tag_idx in range(num_tags):
            tag_embedding = self.tag_embeddings[tag_idx]
            # Compute cosine similarity
            similarity = np.dot(word_embedding, tag_embedding) / (
                        np.linalg.norm(word_embedding) * np.linalg.norm(tag_embedding) + 1e-12)
            emission_scores[tag_idx] = similarity

        # Convert scores to probabilities
        emission_probs = np.exp(emission_scores)
        emission_probs /= np.sum(emission_probs)

        return emission_probs

    def compute_tag_embeddings(self):
        """
        Computes average embeddings for each tag based on words in the training data.
        """
        num_tags = len(self.all_tags)
        self.tag_embeddings = np.zeros((num_tags, self.embedding_dim))
        tag_counts = np.zeros(num_tags)

        for word, idx in self.word2idx.items():
            if word in self.glove_embeddings:
                word_embedding = self.glove_embeddings[word]
                for tag_idx in range(num_tags):
                    count = self.emission_counts[tag_idx, idx]
                    if count > 0:
                        self.tag_embeddings[tag_idx] += word_embedding * count
                        tag_counts[tag_idx] += count

        for tag_idx in range(num_tags):
            if tag_counts[tag_idx] > 0:
                self.tag_embeddings[tag_idx] /= tag_counts[tag_idx]
            else:
                self.tag_embeddings[tag_idx] = np.zeros(self.embedding_dim)

    def inference(self, sequence, method='beam'):
        """Tags a sequence with part of speech tags."""
        if method == 'greedy':
            return self.greedy_decode(sequence)
        elif method == 'beam':
            return self.beam_search_decode(sequence)
        elif method == 'viterbi':
            return self.viterbi_decode(sequence)
        else:
            raise ValueError('Unknown decoding method: {}'.format(method))

    def greedy_decode(self, sequence):
        """
        Greedy decoding implementation.
        """
        num_tags = len(self.all_tags)
        tag_sequence = []
        prev_prev_tag_idx = None
        prev_tag_idx = None

        for i, word in enumerate(sequence):
            word_idx = self.word2idx.get(word, None)
            if word_idx is None:
                # Handle unknown word
                emission_probs = self.handle_unknown_word(word)
            else:
                # Get emission probabilities for this word
                emission_probs = self.emission_probs[:, word_idx]

            max_prob = 0
            best_tag_idx = None

            for tag_idx in range(num_tags):
                # Compute transition probability
                if prev_prev_tag_idx is not None and prev_tag_idx is not None:
                    trans_prob = self.transition_probs[prev_prev_tag_idx, prev_tag_idx, tag_idx]
                elif prev_tag_idx is not None:
                    # Use bigram probability
                    trans_prob = self.l2 * self.bigram_probs_ml[prev_tag_idx, tag_idx] + \
                                 self.l3 * self.unigram_probs_ml[tag_idx]
                else:
                    # Use unigram probability
                    trans_prob = self.unigram_probs_ml[tag_idx]

                # Compute total probability
                prob = trans_prob * emission_probs[tag_idx]

                if prob > max_prob:
                    max_prob = prob
                    best_tag_idx = tag_idx

            tag_sequence.append(self.idx2tag[best_tag_idx])

            # Update previous tags
            prev_prev_tag_idx = prev_tag_idx
            prev_tag_idx = best_tag_idx

        return tag_sequence

    def beam_search_decode(self, sequence, beam_width=3):
        """
        Beam search decoding implementation.
        
        :param sequence: The input sequence of words to tag.
        :param beam_width: The number of top paths to retain at each step (k-best paths).
        :return: The best tag sequence as a list of tag names.
        """
        num_tags = len(self.all_tags)
        
        # Each element in the beam is a tuple of (path, log_probability)
        # `path` is a list of tag indices, and `log_probability` is the accumulated log probability of that path.
        beam = [([], 0.0)]  # Start with an empty path with log probability 0 (log(1))
        
        for i, word in enumerate(sequence):
            word_idx = self.word2idx.get(word, None)
            if word_idx is None:
                # Handle unknown word
                emission_probs = self.handle_unknown_word(word)
            else:
                # Get emission probabilities for this word
                emission_probs = self.emission_probs[:, word_idx]
            
            # To store all potential paths from the current beam
            new_beam = []
            
            # Expand each path in the beam
            for path, log_prob in beam:
                # Last two tags in the current path
                prev_prev_tag_idx = path[-2] if len(path) > 1 else None
                prev_tag_idx = path[-1] if len(path) > 0 else None
                
                # Consider all possible next tags
                for tag_idx in range(num_tags):
                    # Compute transition probability
                    if prev_prev_tag_idx is not None and prev_tag_idx is not None:
                        trans_prob = self.transition_probs[prev_prev_tag_idx, prev_tag_idx, tag_idx]
                    elif prev_tag_idx is not None:
                        # Use bigram probability
                        trans_prob = self.l2 * self.bigram_probs_ml[prev_tag_idx, tag_idx] + \
                                    self.l3 * self.unigram_probs_ml[tag_idx]
                    else:
                        # Use unigram probability
                        trans_prob = self.unigram_probs_ml[tag_idx]
                    
                    # Avoid log(0) by setting a minimum probability
                    trans_prob = max(trans_prob, 1e-12)
                    emission_prob = emission_probs[tag_idx]
                    emission_prob = max(emission_prob, 1e-12)
                    
                    # Compute total log probability
                    total_log_prob = log_prob + math.log(trans_prob) + math.log(emission_prob)
                    
                    # Add the new path with its updated log probability
                    new_beam.append((path + [tag_idx], total_log_prob))
            
            # Sort the new beam by log probability and retain only the top `beam_width` paths
            new_beam.sort(key=lambda x: x[1], reverse=True)  # Higher log_prob is better
            beam = new_beam[:beam_width]
        
        # Select the best path from the beam (highest log probability)
        best_path, best_log_prob = max(beam, key=lambda x: x[1])
        
        # Convert tag indices to tag names
        tag_sequence = [self.idx2tag[tag_idx] for tag_idx in best_path]
        
        return tag_sequence



    def viterbi_decode(self, sequence):
        """
        Viterbi decoding implementation.
        """
        return

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
    for idx, tag in enumerate(test_predictions):
        # Ensure the tag is properly quoted
        results.append({'id': idx, 'tag': str(tag)})

    # Create a DataFrame with the predictions
    df_predictions = pd.DataFrame(results)

    # Write them to a file to update the leaderboard
    df_predictions.to_csv('predictions.csv', index=False, quoting=csv.QUOTE_ALL)
