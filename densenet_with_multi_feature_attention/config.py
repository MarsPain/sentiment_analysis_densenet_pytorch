import os


ckpt_dir = "ckpt"
train_data_path = "../data/sentiment_analysis_trainingset.csv"
dev_data_path = "../data/sentiment_analysis_validationset.csv"
pkl_dir = "pkl"
tfidf_dict_path = "../data/tfidf.txt"
use_pretrained_embedding = True
word2vec_model_path = "../data/word2vec_word_model_sg.txt"
decay_lr_flag = True
column_index = 2
num_classes = 4
num_epochs = 30
batch_size = 32
vocab_size = 210000
tokenize_style = "word"
embed_size = 100
num_filters = 32
num_dense_layer = 5
max_len = 501   # 必须是top_k的倍数
top_k = 3
learning_rate = 0.001
clip_gradients = 3.0
validate_every = 1
dropout_rate = 0.5
