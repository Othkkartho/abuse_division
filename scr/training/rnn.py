# 전처리 코드 불러오기
from scr.training.pretreatment import pretreatment, sentiment_predict
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class RNN:
    def BiLstmLearn(self, embedding_dim, hidden_units, vocab_size, X_train, X_test, y_train, y_test):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(Bidirectional(LSTM(hidden_units)))  # Bidirectional LSTM을 사용
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('../../data/model/rnn_BiLSTM_best_model.h5', monitor='val_acc', mode='max', verbose=1,
                             save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=4, callbacks=[es, mc], batch_size=64, validation_split=0.2)

        loaded_model = load_model('../../data/model/rnn_BiLSTM_best_model.h5')
        print("테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

        return loaded_model

    # LSTM으로 네이버 영화 리뷰 감성 분류하기

    def LstmLearn(self, embedding_dim, hidden_units, vocab_size, X_train, X_test, y_train, y_test):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(LSTM(hidden_units))
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('../../data/model/rnn_LSTM_best_model.h5', monitor='val_acc', mode='max', verbose=1,
                             save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=4, callbacks=[es, mc], batch_size=64, validation_split=0.2)

        loaded_model = load_model('../../data/model/rnn_LSTM_best_model.h5')
        print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

        return loaded_model

    # 리뷰 예측해보기

    def __init__(self):
        embedding_dim = 32
        hidden_units = 32

        X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer = pretreatment()

        loaded_model_lstm = self.LstmLearn(embedding_dim, hidden_units, vocab_size, X_train, X_test, y_train, y_test)
        loaded_model_bilstm = self.BiLstmLearn(embedding_dim, hidden_units, vocab_size, X_train, X_test, y_train, y_test)

        sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해", loaded_model_lstm, stopwords, tokenizer, max_len)
        sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?", loaded_model_lstm, stopwords, tokenizer, max_len)


rnn = RNN()
