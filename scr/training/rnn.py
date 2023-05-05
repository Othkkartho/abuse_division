from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 전처리 코드 불러오기
from scr.training.pretreatment import pretreatment

X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer = pretreatment()

# BiLSTM으로 욕설 데이터 분류하기 분류하기

import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 32
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units)))  # Bidirectional LSTM을 사용
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('../../data/model/rnn_BiLSTM_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('../../data/model/rnn_BiLSTM_best_model.h5')
print("테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# LSTM으로 네이버 영화 리뷰 감성 분류하기

# embedding_dim = 32
# hidden_units = 32
#
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim))
# model.add(LSTM(hidden_units))
# model.add(Dense(1, activation='sigmoid'))
#
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('../../data/model/rnn_LSTM_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train, y_train, epochs=4, callbacks=[es, mc], batch_size=64, validation_split=0.2)
#
# loaded_model = load_model('../../data/model/rnn_LSTM_best_model.h5')
# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


# 리뷰 예측해보기

mecab = Mecab("C:/mecab/mecab-ko-dic")

def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence = mecab.morphs(new_sentence)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(pad_new))  # 예측
    if (score > 0.5):
        print("{:.2f}% 확률로 옥설입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 욕설이 아닙니다.".format((1 - score) * 100))

sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해")
sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?")
