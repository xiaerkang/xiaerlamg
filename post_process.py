# coding = utf-8
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM,Dense,Embedding, Conv1D, Input
#import process_data
import pickle
from crf_keras import CRF
embedding_file = r'C:\Users\PPD\Desktop\data\change\ner/process_data/char_embedding_matrix.npy'

def get_char_tag_data(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        list_all = f.readlines() # type: list
    # print(list_all, len(list_all))
    # ['本 O\n', '性 O\n', '的 O\n', '差 O\n', '别 O\n', '。 O\n', '\n'] 112188

    i = 0
    char_str = str()
    char_list = [] # 列表中每个元素为每句话组成的字符串
    tag_str = str()
    tag_list = [] # 列表中的每个元素为每句话对应的tag所组成的字符串
    while i < len(list_all)-1:
        str_all = list_all[i]
        # print(str_all)
        tep_list = str_all.split("\t")
        #print(tep_list)
        #exit()
        if (len(tep_list) > 1) & (tep_list[0] not in '!。?;'):
            char_str += (tep_list[0] + ' ')
            tag_str += tep_list[1]
        else:
            if tep_list[0] in '!。?;':
                char_str += (tep_list[0] + ' ')
                tag_str += tep_list[1]
            char_list.append(char_str)
            tag_list.append(tag_str)
            char_str = str()
            tag_str = str()
        i += 1
    # print(char_list[:3], tag_list[:3])
    #print(char_list)
    #exit()
    char_data = [sent.split() for sent in char_list if len(sent) > 0] # 将每句话转化为由单字符字符串构成的列表
    tag_data = [tags.split('\n')[:-1] for tags in tag_list if len(tags) > 0] # 同上, 专门去掉''
    #print(len(char_data))
    #print(len(tag_data))
    #print(char_data)
    #exit()
    # 'O\nLOC\nO\n'.split('\n') : ['O', 'LOC', 'O', '']    !!!!
    return char_data, tag_data

def get_char2object():
    #返回词向量的表，chartoid idtochar vocab_size n_embed
    #
    char2vec = {}
    f = open(r'C:\Users\PPD\Desktop\data\change\ner/process_data/wiki_100',encoding="utf8") # load pre-trained word embedding
    i = 0
    for line in f:
        tep_list = line.split()
        if i == 0:
            n_char = int(tep_list[0])
            n_embed = int(tep_list[1])
        else:
            char = tep_list[0]
            vec = np.asarray(tep_list[1:], dtype='float32')
            char2vec[char] = vec
        i += 1
    f.close()
    char2index = {k: i for i, k in enumerate(sorted(char2vec.keys()), 1)}
    #print(n_char)
    #exit()
    return char2vec, n_char, n_embed, char2index

def get_embedding_matrix(char2vec, n_vocab, n_embed, char2index):
    embedding_mat = np.zeros([n_vocab, n_embed])
    #print(char2vec.get("金"))
    #exit()
    for w, i in char2index.items():
        vec = char2vec.get(w)
        #print(vec.shape)
        if vec is not None:
            embedding_mat[i] = vec
    if not os.path.exists(embedding_file):
        np.save(embedding_file, embedding_mat)
    return embedding_mat

def get_X_data(char_data, char2index, max_length):
    index_data = []
    for l in char_data:
        index_data.append([char2index[s] if char2index.get(s) is not None else 0
                           for s in l])
    #print(index_data[1])
    index_array = pad_sequences(index_data, maxlen=max_length, dtype='int32',
                                padding='post', truncating='post', value=0)
    return index_array

def process_data(string, char2index,max_length=200):
    char_data=[]
    length=len(string)
    [char_data.append(i) for i in string]  # 将每句话转化为由单字符字符串构成的列表
    index_data=[]
    [index_data.append(char2index[s]) if char2index.get(s) is not None else 0
     for s in char_data]
    index_data=[index_data]

    index_array = pad_sequences(index_data, maxlen=max_length, dtype='int32',
                                padding='post', truncating='post', value=0)

    return index_array


def get_y_data(tag_data, label2index, max_length):
    index_data = []
    for l in tag_data:
        #print(l)
        index_data.append([label2index[s] for s in l])
        #print(index_data)
        #exit()
    #对数组进行填充
    index_array = pad_sequences(index_data, maxlen=max_length, dtype='int32',
                                padding='post', truncating='post', value=0)
    index_array = to_categorical(index_array, num_classes=7) # (20863, 574, 7)

    # return np.expand_dims(index_array, -1)
    return index_array

if __name__ == '__main__':


    char_train, tag_train = get_char_tag_data(r'C:\Users\PPD\Desktop\data\change\ner/process_data/train.txt')
    char_dev, tag_dev = get_char_tag_data(r'C:\Users\PPD\Desktop\data\change\ner/process_data/dev.txt')
    char_test, tag_test = get_char_tag_data(r'C:\Users\PPD\Desktop\data\change\ner/process_data/test.txt')
    # print(char_train[:3], tag_train[:3])
    char2vec, n_char, n_embed, char2index = get_char2object()
    n_vocab = n_char + 1
    #print(n_embed)
    #exit()
    # print(word2vec['的'], word2index['的']) # n_embed = 100
    if os.path.exists(embedding_file):
        embedding_mat = np.load(embedding_file)
    else:
        embedding_mat = get_embedding_matrix(char2vec, n_vocab, n_embed, char2index)
    #print(embedding_mat)
    #exit(embedding_mat.shape)
    # length = []
    # for data in [char_train, char_dev]:
    #     for l in data:
    #         length.append(len(l))
    # print(max(length), length[800:1000]) # 574
    # count = 0
    # for k in length:
    #     if k > 200:
    #         count += 1
    # print(count, len(length)) # 69 23509
    #print(char_train)

    X_train = get_X_data(char_train, char2index, 200)

    X_dev = get_X_data(char_dev, char2index, 200)
    X_test = get_X_data(char_test, char2index, 200)
    print(X_train.shape, X_dev.shape, X_test.shape) # (21147, 200) (2362, 200) (4706, 200)

    # tag_set = set()
    # for data in [tag_train, tag_dev, tag_test]:
    #     for l in data:
    #         tag_set.update(l)
    # print(tag_set) # {'B-LOC', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O', 'B-ORG'}
    label2index = dict()

    idx = 0
    for c in ['O', 'B', 'I']:
        label2index[c] = idx
        idx += 1
    # print(label2index)

    y_train = get_y_data(tag_train, label2index, 200)
    y_dev = get_y_data(tag_dev, label2index, 200)
    y_test = get_y_data(tag_test, label2index, 200)
    #print(y_train[1])
    sequence = Input(shape=(None,), dtype='int32')  # 建立输入层，输入长度设为None
    embedding = Embedding(n_char,
                          n_embed,
                          )(sequence)  # 去掉了mask_zero=True
    cnn = Conv1D(128, 3, activation='relu', padding='same')(embedding)
    cnn = Conv1D(128, 3, activation='relu', padding='same')(cnn)
    cnn = Conv1D(128, 3, activation='relu', padding='same')(cnn)
    crf = CRF()
    # pred = crf(inputs=[embedding, sequence])

    tag_score = Dense(3)(cnn)  # 变成了5分类，第五个标签用来mask掉
    tag_score = crf(tag_score)  # 包装一下原来的tag_score
    model = Model(inputs=sequence, outputs=tag_score)
    # model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    # model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    # model.add(layers.Conv1D(128, 3, activation='relu', padding='same'))
    # model.add(layers.Conv1D(128, 3, activation='relu', padding='same'))
    # model.add(layers.Conv1D(128, 3, activation='relu', padding='same'))
    # 定义CRF层
    # crf = CRF(True)  # 定义crf层，参数为True，自动mask掉最后一个标签
    # model.add(layers.
    # Model(inputs=sequence, outputs=tag_score)
    # model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss, metrics=[crf.accuracy])
    checkpoint_dir = r'C:\Users\PPD\Desktop\data\change\ner\process_data/model/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    EPOCHS=20
    model.fit(X_train, y_train, batch_size=2024, epochs=EPOCHS,
            callbacks=[checkpoint_callback] ,
           validation_data=[X_test, y_test])
    predict_text = '揭秘趣步骗局，趣步是什么，趣步是怎么赚钱的？趣步公司可靠吗？趣步合法吗'
    predict_text = process_data(predict_text, char2index)
    raw = model.predict(predict_text)[0]
    print(raw)
    exit()
    result = [np.argmax(row) for row in raw]

    result_tags = [chunk_tags[i] for i in result]

    per, loc, org = '', '', ''

    for s, t in zip(predict_text, result_tags):
        if t in ('B-PER', 'I-PER'):
            per += ' ' + s if (t == 'B-PER') else s
        if t in ('B-ORG', 'I-ORG'):
            org += ' ' + s if (t == 'B-ORG') else s
        if t in ('B-LOC', 'I-LOC'):
            loc += ' ' + s if (t == 'B-LOC') else s

    print(['person:' + per, 'location:' + loc, 'organzation:' + org])
    #np.save('data/X_train.npy', X_train)
    #np.save('data/X_dev.npy', X_dev)
    #np.save('data/X_test.npy', X_test)
    #np.save('data/y_train.npy', y_train)
    #np.save('data/y_dev.npy', y_dev)
    #np.save('data/y_test.npy', y_test)