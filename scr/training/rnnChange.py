from scr.training.pretreatment import pretreatment, sentiment_predict, model_loss
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


class RNN:
    def BiLstmLearn(embedding_dim, hidden_units, vocab_size, X_train, X_test, y_train, y_test, batch_size):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(Bidirectional(LSTM(hidden_units)))  # Bidirectional LSTM을 사용
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('../../data/model/rnn_BiLSTM_best_model.h5', monitor='val_acc', mode='max', verbose=1,
                             save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=5, callbacks=[es, mc], batch_size=batch_size, validation_split=0.2)

        loaded_model = load_model('../../data/model/rnn_BiLSTM_best_model.h5')
        loss, final_acc = loaded_model.evaluate(X_test, y_test)
        print("테스트 정확도: %.4f" % final_acc)

        # model_loss(history)

        return loaded_model, loss, final_acc

    # LSTM으로 욕설 분류하기

    def LstmLearn(embedding_dim, hidden_units, vocab_size, X_train, X_test, y_train, y_test, batch_size):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(LSTM(hidden_units))
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('../../data/model/rnn_LSTM_best_model.h5', monitor='val_acc', mode='max', verbose=1,
                             save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=5, callbacks=[es, mc], batch_size=batch_size, validation_split=0.2)

        loaded_model = load_model('../../data/model/rnn_LSTM_best_model.h5')
        loss, final_acc = loaded_model.evaluate(X_test, y_test)
        print("\n 테스트 정확도: %.4f" % final_acc)

        # model_loss(history)

        return loaded_model, loss, final_acc

    # 리뷰 예측해보기

    def start(ci, cj, ck):
        result = []
        embedding_dim = ci
        hidden_units = cj
        batch_size = ck

        result.append(embedding_dim)
        result.append(hidden_units)
        result.append(batch_size)

        X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer = pretreatment()

        start = time.time()
        loaded_model_lstm, l_loss, final_acc_l = RNN.LstmLearn(embedding_dim, hidden_units, vocab_size, X_train, X_test,
                                                        y_train, y_test, batch_size)
        loaded_model_bilstm, b_loss, final_acc_b = RNN.BiLstmLearn(embedding_dim, hidden_units, vocab_size, X_train, X_test,
                                                            y_train, y_test, batch_size)
        end = time.time() - start

        result.append(l_loss)
        result.append(round(final_acc_l*100, 2))
        result.append(b_loss)
        result.append(round(final_acc_b*100, 2))
        result.append(round(end, 3))

        # sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해", loaded_model_lstm, stopwords, tokenizer, max_len)
        # sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?", loaded_model_lstm, stopwords, tokenizer, max_len)

        return result


rnn = []
range_list = [8, 64, 256]
for embedding_dim in range_list:
    for hidden_units in range_list:
        for batch_size in range_list:

            print(embedding_dim, hidden_units, batch_size)

            rnn.append(RNN.start(embedding_dim, hidden_units, batch_size))

for lists in rnn:
    print("embedding_dim: " + str(lists[0]) + ", hidden_units: " + str(lists[1]) + ", batch_size: " + str(lists[2]) +
          ", LSTM loss: " + str(lists[3]) + ", LSTM Acc: " + str(lists[4]) + ", BiLSTM loss: " + str(lists[5]) + ", BiLSTM Acc: " + str(lists[6]) + ", Time Taken: " + str(lists[7]) + "second")
