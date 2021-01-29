from __future__ import absolute_import
from __future__ import print_function

from data_utils import vectorize_data, load_paths

from data_utils import display_data, nell_eval, eval_mrr
from data_utils import get_model_answers, get_real_answers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from itertools import chain
from six.moves import range, reduce
import pandas as pd 
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, RepeatVector
from keras.layers import Lambda, Permute, Dropout, add, multiply, dot
from keras.layers import LSTM, Conv1D, GRU, BatchNormalization
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Reshape
from keras.layers import MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import AveragePooling1D, Bidirectional
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import sequence
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import math
import sys
np.random.seed(7)

# parameters 정의
BATCH_SIZE = int(sys.argv[6])
NUM_HOPS = 2
NUM_EPOCHS = int(sys.argv[7])
EMBEDDING_DIM = 100
LSTM_HIDDEN_UNITS = int(sys.argv[3])
MEMORY_SIZE = 5000
NUM_FILTERS = int(sys.argv[4])
KERNEL_SIZE = 3
MAX_SENT_LENGTH = 10
LEARNING_RATE = float(sys.argv[8])
NUM_TASKS = 1
POOLING_SIZE = int(sys.argv[5])
VOCAB_SIZE = 1000
PATIENCE = int(sys.argv[9])
task = sys.argv[1]

# Data directory 정의
main_dir = os.getcwd()+'/'
processed_data_dir = main_dir+"data/processed_data/" 
nell_dir = main_dir+'data/NELL-995/tasks/concept_'
fb_dir = main_dir+'data/FB15k-237/tasks/'
kinship_dir = main_dir+'data/kinship/tasks/'
countries_dir = main_dir+'data/Countries/tasks/'
task_dir = ''
if sys.argv[2].lower() == "fb15k-237":
    task_dir = fb_dir
    task_uri = '/' + task.replace("@", '/')
elif sys.argv[2].lower() == "nell-995":
    task_dir = nell_dir
    task_uri = task
elif sys.argv[2].lower() == "kinship":
    task_dir = kinship_dir
    task_uri = task
elif sys.argv[2].lower() == "countries":
    task_dir = countries_dir
    task_uri = "locatedin"

tasks = [task]

# train, test DataFrame 정의
label = ['context', 'e1', 'r', 'e2', 'label']
train = pd.DataFrame(columns=label)
test = pd.DataFrame(columns=label)

# pre-processing 된 train, test data를 DataFrame 형식으로 변환
for task in tasks:
    print('task name:', task)
    train0, test0 = load_paths(processed_data_dir + task)
    train = pd.concat([train, train0])
    test = pd.concat([test, test0])

# NELL-995 data의 parsing
# input : line, output : parsing line
def clean_nell(line):
    return line.replace('\n', '').replace('concept:', '').replace('thing$', '').replace("concept_", '')

# 정렬된 test data, correct train data, corrupt train data, 
# correct test data, corrupt test data를 담을 list 정의
sort_test = []
train_pos = []
train_neg = []
test_pos = []
test_neg = []

# 정의한 tasks를 반복하면서 모델 학습과 테스트에 필요한 data들을 정의
for task in tasks:
    # Correct train data 정의하기
    with open(task_dir + task + '/train_pos', 'r') as f:
        for line in f:
            e1, e2, r = clean_nell(line).lower().split('\t') # e1 : subject, e2 : object, r : relation
            # 중복을 제거하기 위해 train_pos에 없을 시 해당 라인을 append
            if (e1, r, e2) not in train_pos:
                train_pos.append((e1, r, e2))
    
    # Corrupt train data 정의하기         
    with open(task_dir + task + '/train.pairs', 'r') as f:
        for line in f:
            pair, l = clean_nell(line).lower().split(': ') # pair : subject/obect, l : label
            # comma로 구분되어 있는 pair를 split 하여 subject, object로 분리
            e1, e2 = pair.split(',')
            # lable이 -인 라인만 train_neg에 append
            if (e1, task_uri.lower(), e2) not in train_neg and l == '-':
                train_neg.append((e1, task, e2))

    # 정렬된 test data 정의하기         
    with open(task_dir + task + '/sort_test.pairs', 'r') as f:
        for line in f:
            pair, l = clean_nell(line).lower().split(': ') # pair : subject/object, l:label
            e1, e2 = pair.split(',')
            e1_uri = e1.lower()
            e2_uri = e2.lower()
            # FB15K-237 data라면 parsing
            if sys.argv[2].lower() == "fb15k-237":
                e1_uri = '/' + e1.replace('_', '/')
                e2_uri = '/' + e2.replace('_', '/')
            # 모든 test data를 sort_test에 append 하되 correct data이면 1, corrupt data는 0으로 labeling
            sort_test.append([e1_uri + '' + task_uri, e2_uri, 1 if l == '+' else 0])
            # correct test data는 test_pos에 append
            if l == '+':
                test_pos.append((e1_uri, task_uri, e2_uri))
            # corrupt test data는 test_neg에 append
            else:
                test_neg.append((e1_uri, task_uri, e2_uri))


# Traing data 중 잘못 labeling 된 data를 filtering 해주는 함수
def filter_overlap_train(x):
    sample = (x['e1'], x['r'], x['e2'])
    if x['label'] == '-':        
        return sample not in train_pos
    else:        
        return sample not in train_neg

# train data filtering
# '+'라고 labeling 된 data 중 corrupt data가 있다면 filtering
# '-'라고 labeling 된 data 중 correct data가 있다면 filtering
train = train[train.apply(filter_overlap_train, axis=1)]

# Test data 중 잘못 labeling 된 data를 filtering 해주는 함수
def filter_overlap_test(x):
    sample = (x['e1'], x['r'], x['e2'])
    if x['label'] == '-':
        return sample not in test_pos
    else:        
        return True #if sample not in test_neg 

# test data filtering
# '+'라고 labeling 된 data 중 corrupt data가 있다면 filtering
# '-'라고 labeling 된 data 중 correct data가 있다면 filtering
test = test[test.apply(filter_overlap_test, axis=1)]
     

data = pd.concat([train, test]) # train, test data를 결합한 모든 data 정의
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + [r]) for s, r in data[['context', 'r']].values.tolist()))) # vocab 정의
word2idx = dict((e, i + 1) for i, e in enumerate(vocab)) # entities와 index를 매핑시킨 dictionary 정의
MAX_SENTS = max(map(len, data['context'].values.tolist())) # 가장 많은 paths의 갯수를 정의
MAX_SENT_LENGTH = max(map(len, chain.from_iterable(data['context'].values))) # 가장 긴 paths의 길이를 정의
MEMORY_SIZE = min(MEMORY_SIZE, MAX_SENTS) # memory size 정의
VOCAB_SIZE = len(word2idx) + 1 # entities 갯수 정의

# text로 표현된 train data를 index로 변환
S, R, L = vectorize_data(train, word2idx, MAX_SENT_LENGTH, MEMORY_SIZE)
# train, validation data로 분리
trainS, valS, trainR, valR, trainL, valL = train_test_split(S, R, L, test_size=.3, random_state=None)
# text로 표현된 test data를 index로 변환
testS, testR, testL = vectorize_data(test, word2idx, MAX_SENT_LENGTH, MEMORY_SIZE)

n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)
print("Entity&Relation Vocab Size", VOCAB_SIZE)

# CNN과 BiLSTM을 결합한 Embedding 모듈
def sentEncoder(embedding):
    sent_model = Sequential()
    sent_model.add(embedding)
    sent_model.add(Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='same', activation='relu')) # 1차원 Convolutional 필터 적용
    sent_model.add(MaxPooling1D(pool_size=POOLING_SIZE)) # Max pooling Layer
    sent_model.add(Bidirectional(LSTM(int(EMBEDDING_DIM/2)))) # Bidirectional LSTM Layer
    return sent_model

# Attention 모듈
def ConvBiLSTM(step):
    query = Input(shape=(MAX_SENT_LENGTH,)) # relation token
    context = Input(shape=(MEMORY_SIZE, MAX_SENT_LENGTH,)) # path sentence tokens
    embedding_A = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SENT_LENGTH, mask_zero=False) # embedding Layer
    query_encoded = sentEncoder(embedding_A)(query) # relation embedding
    context_encoded = TimeDistributed(sentEncoder(embedding_A))(context) # path embeddings
    
    u = query_encoded
    for k in range(step):
        # compare relation embedding and path embeddings
        u_rep = RepeatVector(MEMORY_SIZE)(u)

        ##### tanh similarity function / dot ##############
        output = concatenate([context_encoded, u_rep])
        tanh = Dense(EMBEDDING_DIM, activation='tanh')(output)
        # path attention scores
        score = Dense(1)(tanh)
        
        # probabilistic score
        score = Flatten()(score)
        alpha = Activation('softmax')(score)

        # weighted sum
        o = dot([alpha, context_encoded], axes=(1,1))
        u = add([o, u]) # concenate [o, u] when step is 1
        u = Dense(EMBEDDING_DIM, input_shape=(EMBEDDING_DIM,))(u) # fully connected layer
    
    u = Dense(int(EMBEDDING_DIM), activation='relu')(u) # fully connected layer
    u = Dense(int(EMBEDDING_DIM/2), activation='relu')(u) # fully connected layer

    prediction = Dense(1, activation='sigmoid')(u) # sigmoid layer
    model = Model([context, query], prediction)
    return model

# 모델 정의
model = ConvBiLSTM(2)

# 모델 optimizer 및 loss function 정의
adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

trainR1 = sequence.pad_sequences(trainR.reshape([trainR.shape[0], 1]), MAX_SENT_LENGTH)
valR1 = sequence.pad_sequences(valR.reshape([valR.shape[0], 1]), MAX_SENT_LENGTH)
testR1 = sequence.pad_sequences(testR.reshape([testR.shape[0], 1]), MAX_SENT_LENGTH)

# EarlyStopping 설정 및 학습된 wiehgt 저장 경로 지정
callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE),
                ModelCheckpoint(filepath='output/' + task + '_model.h5', monitor='val_loss', save_best_only=True)]

# Training 시작
history = model.fit([trainS, trainR1], trainL, 
                    validation_data=([valS, valR1], valL),
                    callbacks=callbacks,
                    epochs=NUM_EPOCHS, 
                    batch_size=BATCH_SIZE, verbose=1) 

# 학습된 weight 불러오기
model.load_weights('output/' + task + '_model.h5')

# Testing
test_preds = model.predict([testS, testR1])
model_answers, real_answers = get_model_answers(test_preds, test.values.tolist())

# MAP, MRR, Hits@1, Hits@3, Hits@10 계산 및 출력
mean_ap = nell_eval(model_answers, sort_test)                    
mrr, hits_at1, hits_at3, hits_at10 = eval_mrr(model_answers, sort_test)
print("map:", mean_ap)
print("mrr:", mrr)
print("hits_at1:", hits_at1)
print("hits_at3:", hits_at3)
print("hits_at10:", hits_at10)

# 해당 relation의 log file 저장
log_file = sys.argv[10] 
with open(log_file, 'a') as f:
    f.write('relation: ' + task + '\n')
    f.write('MAP:' + str(mean_ap) + '\n')
    f.write('MRR:' + str(mrr) + '\n')
    f.write('hits@1:  ' + str(hits_at1) + '\n')
    f.write('hits@3:  ' + str(hits_at3) + '\n')
    f.write('hits@10: ' + str(hits_at10) + '\n')
    f.write('\n')