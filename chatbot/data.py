# Copyright 2019. All Rights Reserved.
import config
import collections
import tqdm  # progress bar
import nltk
import jieba
import pickle
import random
import os
import numpy as np

"""
Chatbot data process.
"""
Pair = collections.namedtuple("Pair", "question answer")


class Batch:
    def __int__(self):
        self.encoder_seqs = []
        self.decoder_seqs = []
        self.target_seqs = []
        self.weights = []


class Data:

    def __init__(self):
        self.conf = config.Config()
        self.pad_token = self.get_word_id(self.conf.pad_token)
        self.start_token = self.get_word_id(self.conf.start_token)
        self.end_token = self.get_word_id(self.conf.end_token)
        self.unknown_token = self.get_word_id(self.conf.unknown_token)

    # 2d array containing each question and his answer [[input,target]]
    pad_token = 0
    start_token = 0
    end_token = 0
    unknown_token = 0
    training_samples = []
    word2id = {}
    id2word = {}
    id_count = collections.defaultdict(int)

    def get_line(self):
        conversations = []
        question_file = open(self.conf.question_path, 'r', encoding='utf-8')
        answer_file = open(self.conf.answer_path, 'r', encoding='utf-8')
        # pair question and answer
        for q, a in zip(question_file, answer_file):
            pair = Pair(q, a)
            conversations.append(pair)
        question_file.close()
        answer_file.close()
        return conversations

    def get_word_id(self, word, create=True):
        word = word.lower()
        if not create:
            word_id = self.word2id.get(word, self.unknown_token)
        else:
            if word in self.word2id:
                word_id = self.word2id[word]
            else:
                word_id = len(self.word2id)
                self.word2id[word] = word_id
                self.id2word[word_id] = word
            self.id_count[word] += 1
        return word_id

    def extract_conversation(self, conversation):
        input_words = self.extract_text(conversation.question)
        target_words = self.extract_text(conversation.answer)
        if input_words and target_words:
            self.training_samples.append([input_words, target_words])

    def extract_text(self, line):
        """
        Sentence to vector.
        :param line: A sentence to be converted.
        :return: A vector.
        """
        tokens = []
        sentence_token = nltk.sent_tokenize(line)  # todo why
        for i in range(len(sentence_token)):
            if self.conf.corpus == 'cn':
                # tokens = list(line)
                tokens = jieba.cut(line)
            elif self.conf.corpus == 'en':
                tokens = nltk.word_tokenize(sentence_token[i])
            else:
                assert False, 'Please type a correct corpus name'
        return [self.get_word_id(token) for token in tokens]

    def save(self):
        conversations = self.get_line()
        for conversation in tqdm.tqdm(conversations, desc='Extract conversations'):
            # print(conversation)
            self.extract_conversation(conversation)
        self.preprocess()
        print(self.conf.filtered_sample_path)

        with open(self.conf.filtered_sample_path, 'wb') as fh:
            data = {'word2id': self.word2id, 'id2word': self.id2word, 'id_count': self.id_count,
                    'training_samples': self.training_samples}
            pickle.dump(data, fh, -1)

    def load(self):
        with open(self.conf.filtered_sample_path, 'rb') as fh:
            data = pickle.load(fh)
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.id_count = data['id_count']
            self.training_samples = data['training_samples']
            return data['word2id'], data['id2word'], data['id_count'], data['training_samples']

    def preprocess(self):
        """copied from teacher"""

        # 对Q或者A，如果数据长度超过maxlength,则停止，如果不超过则合并词到句子里，可以控制是
        # 从前往后也可以从后往前进行截取
        def mergeSentences(sentences, fromEnd=False):
            """Merge the sentences until the max sentence length is reached
            Also decrement id count for unused sentences.
            Args:
                sentences (list<list<int>>): the list of sentences for the current line
                fromEnd (bool): Define the question on the answer
            Return:
                list<int>: the list of the word ids of the sentence
            """
            # We add sentence by sentence until we reach the maximum length
            merged = []

            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if fromEnd:
                sentences = reversed(sentences)
            for sentence in sentences:
                # If the total length is not too big, we still can add one more sentence
                if len(merged) + len(sentence) <= self.conf.max_length:
                    if fromEnd:  # Append the sentence
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:  # If the sentence is not used, neither are the words
                    for w in sentence:
                        self.id_count[w] -= 1
            return merged

        newSamples = []
        # 1 根据指定句子词长度进行过滤
        # 1st step: Iterate over all words and add filters the sentences
        # according to the sentence lengths
        for inputWords, targetWords in tqdm.tqdm(self.training_samples, desc='Filter sentences:', leave=False):
            inputWords = mergeSentences(inputWords, fromEnd=True)
            targetWords = mergeSentences(targetWords, fromEnd=False)
            newSamples.append([inputWords, targetWords])
        words = []

        # WARNING: DO NOT FILTER THE UNKNOWN TOKEN !!! Only word which has count==0 ?
        # 2 过滤低频词
        # 2nd step: filter the unused words and replace them by the unknown token
        # This is also where we update the correnspondance dictionaries
        # TODO: bad HACK to filter the special tokens. Error prone if one day add new special tokens
        specialTokens = {self.pad_token, self.start_token, self.end_token, self.unknown_token}
        newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0
        selectedWordIds = collections.Counter(self.id_count).most_common(
            self.conf.vocabulary_size or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds if v > self.conf.filter_vocab}
        selectedWordIds |= specialTokens

        for wordId, count in [(i, self.id_count[i]) for i in range(len(self.id_count))]:  # Iterate in order
            if wordId in selectedWordIds:  # Update the word id
                newMapping[wordId] = newId
                word = self.id2word[wordId]  # The new id has changed, update the dictionaries
                del self.id2word[wordId]  # Will be recreated if newId == wordId
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else:  # Cadidate to filtering, map it to unknownToken (Warning: don't filter special token)
                newMapping[wordId] = self.unknown_token
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]

        # 3 更新词频id
        # Last step: replace old ids by new ones and filters empty sentences
        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknown_token:  # Also filter if only contains unknown tokens
                    valid = True
            return valid

        self.training_samples.clear()

        for inputWords, targetWords in tqdm.tqdm(newSamples, desc='Replace ids:', leave=False):
            valid = True
            valid &= replace_words(inputWords)
            valid &= replace_words(targetWords)
            # valid &= targetWords.count(self.unknownToken) == 0  # Filter target with out-of-vocabulary target words ?
            if valid:
                self.training_samples.append([inputWords, targetWords])  # TODO: Could replace list by tuple
        self.id_count.clear()  # Not usefull anymore. Free data

    def _create_batch(self, samples):
        batch = Batch()
        batch_size = len(samples)
        for i in range(batch_size):
            sample = samples[i]
            batch.encoder_seqs.append(list(reversed(sample[0])))
            batch.decoder_seqs.append([self.start_token] + sample[1] + [self.end_token])
            batch.target_seqs.append(sample[1] + [self.end_token])  # Same as decoder, but omitting the start_token
            assert len(batch.encoder_seqs[i]) <= self.conf.max_length_encode
            assert len(batch.decoder_seqs[i]) <= self.conf.max_length_encode
            # padding
            batch.encoder_seqs[i] = [self.pad_token] * (self.conf.max_length_encode - len(batch.encoder_seqs[i])) + \
                                    batch.encoder_seqs[i]  # left padding
            batch.decoder_seqs[i] = batch.decoder_seqs[i] + [self.pad_token] * (
                    self.conf.max_length_decode - len(batch.decoder_seqs[i]))  # right padding
            batch.target_seqs[i] = batch.target_seqs[i] + [self.pad_token] * (
                    self.conf.max_length_decode - len(batch.target_seqs[i]))
            batch.weights.append(
                [1.0] * len(batch.target_seqs[i]) + [0.0] * (self.conf.max_length_decode - len(batch.target_seqs[i])))

        return np.transpose(batch)

    def get_batch(self):
        print('Shuffling the data set...')
        random.shuffle(self.training_samples)
        batches = []

        def get_next_sample():
            for i in range(len(self.training_samples), random.randint(12, 20)):
                yield self.training_samples[i:min(i + self.conf.batch_size, len(self.training_samples))]

        for sample in get_next_sample():
            batch = self._create_batch(sample)
            batches.append(batch)
        return batches


if __name__ == '__main__':
    data = Data()
    # data.save()
    *first, training_samples = data.load()
    print(np.array(training_samples).shape)
