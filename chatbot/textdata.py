# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Loads the dialogue corpus, builds the vocabulary
"""
import numpy as np
import nltk  # For tokenize
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
import collections



class Batch:
    """Struct containing batches info
    """

    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:
    # 初始化加载数据集，判断是否数据摸底
    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args
        # Path variables
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        # basePath = self._constructBasePath()
        path = os.path.join(self.args.rootDir, 'data/samples/')
        path += 'dataset-{}'.format(self.args.modelTag)
        if self.args.datasetTag:
            path += '-' + self.args.datasetTag
        basePath = path
        self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(self.args.maxLength,
                                                                                          self.args.filterVocab,
                                                                                          self.args.vocabularySize, )  # Sentences/vocab filtered for this model
        print(self.filteredSamplesPath)
        # 最好改成1234
        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary
        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]
        self.word2id = {}
        self.id2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)
        self.idCount = {}  # Useful to filters the words (TODO: Could replace dict by list or use collections.Counter)
        self.loadCorpus()
        # Plot some stats:
        print('Loaded {}: {} words, {} QA'.format(self.args.corpus, len(self.word2id), len(self.trainingSamples)))

    # 样本创建训练batch数据
    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """
        batch = Batch()
        batchSize = len(samples)
        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]
            if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer, This could move to the data processing area
                sample = list(reversed(sample))
            if not self.args.test and self.args.autoEncode:  # Autoencode: use either the question or answer for both input and output
                k = random.randint(0, 1)
                sample = (sample[k], sample[k])
            # TODO: Why re-processed that at each epoch ? Could precompute that
            # once and reuse those every time. Is not the bottleneck so won't change
            # much ? and if preprocessing, should be compatible with autoEncode & cie.
            batch.encoderSeqs.append(list(reversed(
                sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(
                batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco
            # Pad数据，做定长数据作为输入，maxlength控制序列长度
            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            # 1. encorder的输入：人物1说的一句话A，最大长度10 
            # 2. decoder的输入：人物2回复的对话B，因为前后分别加上了go开始符和end结束符，最大长度为12 
            # 3. decoder的target输入：decoder输入的目标输出，与decoder的输入一样但只有end标示符号，可以理解为decoder的输入在时序上的结果，比如说完这个词后的下个词的结果。 
            # 4. decoder的weight输入：用来标记target中的非padding的位置，即实际句子的长度，因为不是所有的句子的长度都一样，在实际输入的过程中，各个句子的长度都会被用统一的标示符来填充（padding）至最大长度，weight用来标记实际词汇的位置，代表这个位置将会有梯度值回传。
            batch.encoderSeqs[i] = [self.padToken] * (self.args.maxLengthEnco - len(batch.encoderSeqs[i])) + \
                                   batch.encoderSeqs[i]  # Left padding for the input
            batch.weights.append(
                [1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (
                        self.args.maxLengthDeco - len(batch.decoderSeqs[i]))  # Right padding for the decoder input
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.padToken] * (
                        self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        # Simple hack to reshape the batch
        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT
        return batch

    # 为当前epoch准备输入batch
    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        # self.shuffle()
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)
        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, len(self.trainingSamples), random.randint(12, 20)):
                # for i in range(0, len(self.trainingSamples), self.args.batchSize):
                # random.randint(12, 20) it support different batchsize for each train, as placeholder is None dimension
                yield self.trainingSamples[i:min(i + self.args.batchSize, len(self.trainingSamples))]

        # TODO: Should replace that by generator (better: by tf.queue)
        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches

    # 加载数据集，如果数据集不存在，则创建数据集
    def loadCorpus(self):
        """Load/create the conversations data
        """
        print(self.filteredSamplesPath)
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')
        else:
            # self.loadDataset(self.filteredSamplesPath)
            """Load samples from file
            Args:
                filename (str): pickle filename
            """
            filename = self.filteredSamplesPath
            dataset_path = os.path.join(filename)
            print('Loading dataset from {}'.format(dataset_path))
            with open(dataset_path, 'rb') as handle:
                data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                self.word2id = data['word2id']
                self.id2word = data['id2word']
                self.idCount = data.get('idCount', None)
                self.trainingSamples = data['trainingSamples']
                self.padToken = self.word2id['<pad>']
                self.goToken = self.word2id['<go>']
                self.eosToken = self.word2id['<eos>']
                self.unknownToken = self.word2id['<unknown>']  # Restore special words
        # assert self.padToken == 0

    # 获取wordid
    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # At inference, we simply look up for the word
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        # Get the id if the word already exist
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        # If not, we create a new entry
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1
        return wordId

    # 将数字列表转换为字符串
    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """
        if not sequence:
            return ''
        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])
        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])
        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        # 将分词后的词转换为句子，添加分隔
        return ''.join([' ' + t if not t.startswith('\'') and t not in string.punctuation else t for t in
                        sentence]).strip().capitalize()

    # 将批量数字列表转换为字符串
    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    # 将一个string sentence转换为模型batch输入流水线
    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        if self.args.corpus == "cn":
            tokens = list(sentence)
            # import jieba
            # tokens = jieba.cut(sentence)
            print(tokens)
        elif self.args.corpus == "en":
            tokens = nltk.word_tokenize(sentence)
        else:
            assert ("please type correct")
        if len(tokens) > self.args.maxLength:
            return None
        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences
        print(wordIds)
        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output
        return batch
