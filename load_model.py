import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM,Dense,Embedding, Conv1D, Input
from crf_keras import CRF
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model
checkpoint_dir = r'C:\Users\PPD\Desktop\data\change\ner\process_data/model/training_checkpoints'
# Name of the checkpoint files

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
#定义保存模型名称
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
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
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

raw = model.predict(predict_text)[0]
print(raw)
result = [np.argmax(row) for row in raw]