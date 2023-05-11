from scr.training.pretreatment import pretreatment, sentiment_predict, model_loss
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time


class CNN:
    def multiKernel(vocab_size, X_train, X_test, y_train, y_test, max_len, embedding_dim, num_filters, batch_size):
        dropout_ratio = (0.5, 0.8)
        hidden_units = 128

        model_input = Input(shape=(max_len,))
        z = Embedding(vocab_size, embedding_dim, input_length=max_len, name="embedding")(model_input)
        z = Dropout(dropout_ratio[0])(z)

        conv_blocks = []

        for sz in [3, 4, 5]:
            conv = Conv1D(filters=num_filters,
                          kernel_size=sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(z)
            conv = GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)

        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        z = Dropout(dropout_ratio[1])(z)
        z = Dense(hidden_units, activation="relu")(z)
        model_output = Dense(1, activation="sigmoid")(z)

        model = Model(model_input, model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('../../data/model/CNN_mk_model.h5', monitor='val_acc', mode='max', verbose=1,
                             save_best_only=True)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.2, verbose=2,
                            callbacks=[es, mc])

        loaded_model = load_model('../../data/model/CNN_mk_model.h5')
        loss, final_acc = loaded_model.evaluate(X_test, y_test)
        print("\n 테스트 정확도: %.4f" % final_acc)

        # model_loss(history)

        return loaded_model, loss, final_acc

    # 1D CNN으로 욕설 예측해보기

    def onedcnn(vocab_size, X_train, X_test, y_train, y_test, max_len, embedding_dim, num_filters, batch_size):
        dropout_ratio = 0.3
        kernel_size = 5

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(Dropout(dropout_ratio))
        model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(dropout_ratio))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('../../data/model/CNN_od_model.h5', monitor='val_acc', mode='max', verbose=1,
                             save_best_only=True)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.2, verbose=2,
                            callbacks=[es, mc])

        str_X_train = [str(x) for x in X_train]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(str_X_train)

        str_X_test = [str(x) for x in X_test]
        X_test_encoded = tokenizer.texts_to_sequences(str_X_test)
        X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len)

        loaded_model = load_model('../../data/model/CNN_od_model.h5')
        loss, final_acc = loaded_model.evaluate(X_test, y_test)

        print("\n 테스트 정확도: %.4f" % final_acc)

        # model_loss(history)

        return loaded_model, loss, final_acc

    def start(embedding_dim, num_filters, batch_size):
        result = []
        X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer = pretreatment()

        result.append(embedding_dim)
        result.append(num_filters)
        result.append(batch_size)

        max_len = 60

        start = time.time()
        loaded_model_mk, m_loss, final_acc_m = CNN.multiKernel(vocab_size, X_train, X_test, y_train, y_test, max_len, embedding_dim, num_filters, batch_size)
        loaded_model_od, o_loss, final_acc_o = CNN.onedcnn(vocab_size, X_train, X_test, y_train, y_test, max_len, embedding_dim, num_filters, batch_size)
        end = time.time() - start

        result.append(m_loss)
        result.append(final_acc_m)
        result.append(o_loss)
        result.append(final_acc_o)
        result.append(end)

        # sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해", loaded_model_mk, stopwords, tokenizer, max_len)
        # sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해", loaded_model_od, stopwords, tokenizer, max_len)
        # sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?", loaded_model_mk, stopwords, tokenizer, max_len)
        # sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?", loaded_model_od, stopwords, tokenizer, max_len)

        return result


cnn = []
for ci in range(0, 9, 2):
    for cj in range(0, 9, 2):
        for ck in range(0, 9, 2):
            embedding_dim = 2 ** ci
            num_filters = 2 ** cj
            batch_size = 2 ** ck

            print(embedding_dim, num_filters, batch_size)

            cnn.append(CNN.start(embedding_dim, num_filters, batch_size))

for lists in cnn:
    print("embedding_dim: " + str(lists[0]) + ", num_filters: " + str(lists[1]) + ", batch_size: " + str(lists[2]) +
          ", M1D loss: " + str(lists[3]) + ", M1D Acc: " + str(lists[4]) + ", 1D loss: " + str(lists[5]) + ", 1D Acc: " + str(lists[6]), ", Time Taken: " + str(lists[7]))
