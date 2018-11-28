import numpy as np
# from model import TextCNN
import os
import csv
import json
from collections import OrderedDict
import pickle
from densenet_baseline import config
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
from densenet_baseline.data_utils import seg_words, create_dict, shuffle_padding, sentence_word_to_index,\
    get_vector_tfidf, BatchManager, get_max_len, get_weights_for_current_batch, compute_confuse_matrix,\
    get_labal_weight, get_weights_for_current_batch_and_sample, get_sample_weights, get_f_scores_all,\
    get_vector_tfidf_from_dict
from densenet_baseline.utils import load_data_from_csv, get_tfidf_and_save, load_tfidf_dict,\
    load_word_embedding, get_tfidf_dict_and_save, get_idf_dict_and_save
from densenet_baseline.model import DenseNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


filter_sizes = config.filter_sizes
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        self.model_name = None  # 保存模型的文件夹
        self.train_data_df = None   # 训练集
        self.validate_data_df = None    # 验证集
        self.string_train = None    # 训练集的评论字符串
        self.string_valid = None    # 训练集的评论字符串
        self.columns = None  # 列索引的名称
        self.label_train_dict = None  # 用一个字典保存各个评价对象的标签列表
        self.label_valid_dict = None
        self.word_to_index = None   # word到index的映射字典
        self.index_to_word = None   # index到字符word的映射字典
        self.label_to_index = None   # label到index的映射字典
        self.index_to_label = None  # index到label的映射字典
        self.label_weight_dict = None   # 存储标签权重
        self.train_batch_manager = None  # train数据batch生成类
        self.valid_batch_manager = None  # valid数据batch生成类

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-mn', '--model_name', type=str, nargs='?', help='the name of model')
        args = parser.parse_args()
        self.model_name = args.model_name
        if not self.model_name:
            self.model_name = config.ckpt_dir
        if not os.path.isdir(self.model_name):   # 创建存储临时字典数据的目录
            os.makedirs(self.model_name)

    def load_data(self):
        logger.info("start load data")
        self.train_data_df = load_data_from_csv(config.train_data_path)
        self.validate_data_df = load_data_from_csv(config.dev_data_path)
        content_train = self.train_data_df.iloc[:, 1]
        content_valid = self.validate_data_df.iloc[:, 1]
        logger.info("start seg train data")
        if not os.path.isdir(config.pkl_dir):   # 创建存储临时字典数据的目录
            os.makedirs(config.pkl_dir)
        string_train_valid = os.path.join(config.pkl_dir, "string_train_valid.pkl")
        # string_train_valid = os.path.join(config.pkl_dir, "string_train_valid_2.pkl")
        if os.path.exists(string_train_valid):  # 若word_label_path已存在
            with open(string_train_valid, 'rb') as f:
                self.string_train, self.string_valid = pickle.load(f)
        else:
            self.string_train = seg_words(content_train, config.tokenize_style)  # 根据tokenize_style对评论字符串进行分词
            self.string_valid = seg_words(content_valid, config.tokenize_style)
            with open(string_train_valid, 'wb') as f:
                pickle.dump([self.string_train, self.string_valid], f)
        print("训练集大小：", len(self.string_train))
        logger.info("complete seg train data")
        self.columns = self.train_data_df.columns.values.tolist()
        # print(self.columns)
        logger.info("load label data")
        self.label_train_dict = {}
        for column in self.columns[2:]:
            label_train = list(self.train_data_df[column].iloc[:])
            self.label_train_dict[column] = label_train
        self.label_valid_dict = {}
        for column in self.columns[2:]:
            label_valid = list(self.validate_data_df[column].iloc[:])
            self.label_valid_dict[column] = label_valid
        # print(self.label_list["location_traffic_convenience"][0], type(self.label_list["location_traffic_convenience"][0]))

    def get_dict(self):
        logger.info("start get dict")
        if not os.path.isdir(config.pkl_dir):   # 创建存储临时字典数据的目录
            os.makedirs(config.pkl_dir)
        word_label_dict = os.path.join(config.pkl_dir, "word_label_dict.pkl")    # 存储word和label与index之间的双向映射字典
        if os.path.exists(word_label_dict):  # 若word_label_path已存在
            with open(word_label_dict, 'rb') as dict_f:
                self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = pickle.load(dict_f)
        else:   # 重新读取训练数据并创建各个映射字典
            self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = \
                create_dict(self.string_train, self.label_train_dict, word_label_dict, config.vocab_size)
        # print(len(self.word_to_index), self.word_to_index)
        logger.info("complete get dict")

    def get_data(self):
        logger.info("start get data")
        train_valid_test = os.path.join(config.pkl_dir, "train_valid_test.pkl")
        # train_valid_test = os.path.join(config.pkl_dir, "train_valid_test_2.pkl")
        if os.path.exists(train_valid_test):    # 若train_valid_test已被处理和存储
            with open(train_valid_test, 'rb') as data_f:
                train_data, valid_data, self.label_weight_dict = pickle.load(data_f)
        else:   # 读取数据集并创建训练集、验证集
            # 获取tfidf值并存储为tfidf字典
            if not os.path.exists(config.tfidf_dict_path):
                get_tfidf_dict_and_save(self.string_train, config.tfidf_dict_path, config.tokenize_style)
            tfidf_dict = load_tfidf_dict(config.tfidf_dict_path)
            # 根据tfidf_dict获取训练集和验证集的tfidf值向量作为额外的特征向量
            train_vector_tfidf = get_vector_tfidf_from_dict(self.string_train, tfidf_dict)
            valid_vector_tfidf = get_vector_tfidf_from_dict(self.string_valid, tfidf_dict)
            # 获取idf值并存储为idf字典
            # if not os.path.exists(FLAGS.idf_dict_path):
            #     get_idf_dict_and_save(self.string_train, FLAGS.idf_dict_path, config.tokenize_style)
            # idf_dict = load_tfidf_dict(FLAGS.idf_dict_path)
            # # 根据idf_dict获取训练集和验证集的idf值向量作为额外的特征向量
            # train_vector_tfidf = get_vector_tfidf_from_dict(self.string_train, idf_dict)
            # valid_vector_tfidf = get_vector_tfidf_from_dict(self.string_valid, idf_dict)
            # 获取tfidf模型以及已被排序的字典
            # if not os.path.exists(FLAGS.tfidf_path):
            #     vectorizer_tfidf, word_list_sort_dict = get_tfidf_and_save(self.string_train, FLAGS.tfidf_path, config.tokenize_style)
            # else:
            #     with open(FLAGS.tfidf_path, "rb") as f:
            #         vectorizer_tfidf, word_list_sort_dict = pickle.load(f)
            # # 根据tfidf模型以及已被排序的字典获取训练集和验证集的tfidf值向量作为额外的特征向量
            # train_vector_tfidf = get_vector_tfidf(self.string_train, vectorizer_tfidf, word_list_sort_dict)
            # valid_vector_tfidf = get_vector_tfidf(self.string_valid, vectorizer_tfidf, word_list_sort_dict)
            # print(train_vector_tfidf[0])
            # 语句序列化，将句子中的word和label映射成index，作为模型输入
            sentences_train, self.label_train_dict = sentence_word_to_index(self.string_train, self.word_to_index, self.label_train_dict, self.label_to_index)
            sentences_valid, self.label_valid_dict = sentence_word_to_index(self.string_valid, self.word_to_index, self.label_valid_dict, self.label_to_index)
            # print(sentences_train[0])
            # print(self.label_train_dict["location_traffic_convenience"])
            # 打乱数据、padding,并对评论序列、特征向量、标签字典打包
            # max_sentence = get_max_len(sentences_train)  # 获取最大评论序列长度
            train_data = shuffle_padding(sentences_train, train_vector_tfidf, self.label_train_dict, config.max_len)
            valid_data = shuffle_padding(sentences_valid, valid_vector_tfidf, self.label_valid_dict, config.max_len)
            # 从训练集中获取label_weight_dict（存储标签权重）
            self.label_weight_dict = get_labal_weight(train_data[2], self.columns, config.num_classes)
            with open(train_valid_test, "wb") as f:
                pickle.dump([train_data, valid_data, self.label_weight_dict], f)
        print("训练集大小：", len(train_data[0]), "验证集大小：", len(valid_data[0]))
        # 获取train、valid数据的batch生成类
        self.train_batch_manager = BatchManager(train_data, int(config.batch_size))
        print("训练集批次数量：", self.train_batch_manager.len_data)
        self.valid_batch_manager = BatchManager(valid_data, int(config.batch_size))
        logger.info("complete get data")

    def train_control(self):
        """
        控制针对每种评价对象分别进行训练、验证和模型保存，所有模型保存的文件夹都保存在总文件夹ckpt中
        模型文件夹以评价对象进行命名
        :return:
        """
        logger.info("start train")
        column_name_list = self.columns
        column_name = column_name_list[config.column_index]   # 选择评价对象
        logger.info("start %s model train" % column_name)
        self.train(column_name)
        logger.info("complete %s model train" % column_name)
        logger.info("complete all models' train")

    def train(self, column_name):
        model = self.create_model(column_name)
        print("model:", model)
        model.cuda()
        learning_rate = 0.001
        iteration = 0
        best_acc = 0.50
        best_f1_score = 0.20
        for epoch in range(config.num_epochs):
            print("learning_rate:", learning_rate)
            loss, eval_acc, counter = 0.0, 0.0, 0
            input_y_all = []
            predictions_all = []
            # train
            for batch in self.train_batch_manager.iter_batch(shuffle=False):
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                optimizer.zero_grad()
                iteration += 1
                input_x, features_vector, input_y_dict = batch
                input_y = input_y_dict[column_name]
                input_y_all.extend(input_y)
                weights = torch.Tensor(np.asarray(self.label_weight_dict[column_name]))  # 根据类别权重参数更新训练集各标签的权重
                criterion = nn.CrossEntropyLoss(weight=weights.cuda())
                input_x, input_y = torch.Tensor(input_x), torch.Tensor(input_y)
                input_y = input_y.long()
                input_x, input_y = Variable(input_x.cuda()), Variable(input_y.cuda())
                outputs = model(input_x)
                curr_loss = criterion(outputs, input_y)
                _, predictions = torch.max(outputs.data, 1)
                curr_acc = ((predictions.cpu() == input_y.cpu()).sum().numpy()) / len(input_y.cpu())
                predictions_all.extend(predictions.cpu())
                loss, eval_acc, counter = loss+curr_loss.cpu(), eval_acc+curr_acc, counter+1
                if counter % 100 == 0:  # steps_check
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f" % (epoch, counter, loss/float(counter), eval_acc/float(counter)))
                curr_loss.backward()
                optimizer.step()
            f_0, f_1, f_2, f_3 = get_f_scores_all(predictions_all, input_y_all, 0.00001)  # test_f_score_in_valid_data
            print("f_0, f_1, f_2, f_3:", f_0, f_1, f_2, f_3)
            print("f1_score:", (f_0 + f_1 + f_2 + f_3) / 4)
            print("going to increment epoch counter....")
            # valid
            if epoch % config.validate_every == 0:
                eval_y_all = []
                eval_predictions_all = []
                for batch in self.valid_batch_manager.iter_batch(shuffle=True):
                    eval_x, features_vector, eval_y_dict = batch
                    eval_y = eval_y_dict[column_name]
                    eval_y_all.extend(eval_y)
                    eval_x, eval_y = torch.Tensor(eval_x), torch.Tensor(eval_y)
                    eval_y = eval_y.long()
                    eval_x, eval_y = Variable(eval_x.cuda()), Variable(eval_y.cuda())
                    outputs = model(eval_x)
                    _, predictions = torch.max(outputs.data, 1)
                    eval_predictions_all.extend(predictions.cpu())
                f1_scoree, f_0, f_1, f_2, f_3, weights_label = self.evaluate(np.asarray(eval_y_all), np.asarray(eval_predictions_all))
                print("【Validation】Epoch %d\t f_0:%.3f\tf_1:%.3f\tf_2:%.3f\tf_3:%.3f" % (epoch, f_0, f_1, f_2, f_3))
                print("【Validation】Epoch %d\t F1 Score:%.3f" % (epoch, f1_scoree))
                # save model to checkpoint
                if f1_scoree > best_f1_score:
                    save_path = config.ckpt_dir + "/" + column_name + "/model.ckpt"
                    print("going to save model. eval_f1_score:", f1_scoree, ";previous best f1 score:", best_f1_score, ";previous best_acc:", str(best_acc))
                    # torch.save(model, save_path)
                    best_f1_score = f1_scoree
                if config.decay_lr_flag and (epoch != 0 and (epoch == 5 or epoch == 10 or epoch == 15 or epoch == 20)):
                    for i in range(1):  # decay learning rate if necessary.
                        print(i, "Going to decay learning rate by half.")
                        learning_rate = learning_rate / 2

    def create_model(self, column_name):
        model_save_dir = config.ckpt_dir + "/" + column_name
        if os.path.exists(model_save_dir):
            model = torch.load(model_save_dir+"/model.ckpt")
            print("Restoring Variables from Checkpoint.")
        else:
            print('Initializing Variables')
            model = DenseNet()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            # if not os.path.exists(model_save_dir):
            #     os.makedirs(model_save_dir)
            if config.use_pretrained_embedding:  # 加载预训练的词向量
                print("===>>>going to use pretrained word embeddings...")
                emb_matrix = np.zeros((len(self.index_to_word), config.embed_size))
                new_emb_matrix_word2vec = load_word_embedding(emb_matrix, config.word2vec_model_path, config.embed_size, self.index_to_word)
                model.embed.weight.data.copy_(torch.from_numpy(new_emb_matrix_word2vec))
                print("using pre-trained word emebedding.ended...")
        return model

    def evaluate(self, eval_y_all, eval_predictions_all):
        small_value = 0.00001
        true_positive_0_all, false_positive_0_all, false_negative_0_all, true_positive_1_all, false_positive_1_all, false_negative_1_all,\
        true_positive_2_all, false_positive_2_all, false_negative_2_all, true_positive_3_all, false_positive_3_all, false_negative_3_all = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        weights_label = {}  # weight_label[label_index]=(number,correct)
        true_positive_0, false_positive_0, false_negative_0, true_positive_1, false_positive_1, false_negative_1,\
        true_positive_2, false_positive_2, false_negative_2, true_positive_3, false_positive_3, false_negative_3 = compute_confuse_matrix(eval_predictions_all, eval_y_all, small_value)
        true_positive_0_all += true_positive_0
        false_positive_0_all += false_positive_0
        false_negative_0_all += false_negative_0
        true_positive_1_all += true_positive_1
        false_positive_1_all += false_positive_1
        false_negative_1_all += false_negative_1
        true_positive_2_all += true_positive_2
        false_positive_2_all += false_positive_2
        false_negative_2_all += false_negative_2
        true_positive_3_all += true_positive_3
        false_positive_3_all += false_positive_3
        false_negative_3_all += false_negative_3
        # write_predict_error_to_file(file_object, logits, eval_y, self.index_to_word, eval_x1, eval_x2)    # 获取被错误分类的样本（后期再处理）
        # print("标签0的预测情况：", true_positive_0, false_positive_0, false_negative_0)
        p_0 = float(true_positive_0_all)/float(true_positive_0_all+false_positive_0_all+small_value)
        r_0 = float(true_positive_0_all)/float(true_positive_0_all+false_negative_0_all+small_value)
        f_0 = 2 * p_0 * r_0 / (p_0 + r_0 + small_value)
        # print("标签1的预测情况：", true_positive_1, false_positive_1, false_negative_1)
        p_1 = float(true_positive_1_all)/float(true_positive_1_all+false_positive_1_all+small_value)
        r_1 = float(true_positive_1_all)/float(true_positive_1_all+false_negative_1_all+small_value)
        f_1 = 2 * p_1 * r_1 / (p_1 + r_1 + small_value)
        # print("标签2的预测情况：", true_positive_2, false_positive_2, false_negative_2)
        p_2 = float(true_positive_2_all)/float(true_positive_2_all+false_positive_2_all+small_value)
        r_2 = float(true_positive_2_all)/float(true_positive_2_all+false_negative_2_all+small_value)
        f_2 = 2 * p_2 * r_2 / (p_2 + r_2 + small_value)
        # print("标签3的预测情况：", true_positive_3, false_positive_3, false_negative_3)
        p_3 = float(true_positive_3_all)/float(true_positive_3_all+false_positive_3_all+small_value)
        r_3 = float(true_positive_3_all)/float(true_positive_3_all+false_negative_3_all+small_value)
        f_3 = 2 * p_3 * r_3 / (p_3 + r_3 + small_value)
        f1_score = (f_0 + f_1 + f_2 + f_3) / 4
        return f1_score, f_0, f_1, f_2, f_3, weights_label

if __name__ == "__main__":
    main = Main()
    main.get_parser()
    main.load_data()
    main.get_dict()
    main.get_data()
    main.train_control()
