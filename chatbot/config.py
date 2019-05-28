import os


class Config:
    # token
    pad_token = '<pad>'
    start_token = '<start>'
    end_token = '<end>'
    unknown_token = '<unknown>'

    # path
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    question_path = os.path.join(data_dir, 'corpus/question.txt')
    answer_path = os.path.join(data_dir, 'corpus/answer.txt')
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    #
    model_tag = 'hack'
    vocabulary_size = 1000
    corpus = 'cn'
    num_epochs = 30
    max_length = 10
    filter_vocab = 0
    filtered_sample_path = os.path.join(data_dir,
                                        'samples/dataset-{}-{}-filter{}-vacab_size{}.pkl'
                                        .format(model_tag,
                                                max_length,
                                                filter_vocab,
                                                vocabulary_size))

    batch_size = 12
    max_length_encode = max_length
    max_length_decode = max_length + 2
    test = False

    #
    hidden_size = 2
    num_layers = 2
    embedding_size = 64
    learning_rate = 0.002
