# 전처리 코드 불러오기
from scr.training.pretreatment import pretreatment, sentiment_predict, model_loss
from sklearn.model_selection import train_test_split
import autokeras as ak
import numpy as np
import tensorflow as tf


class RNNAutoml:
    # autokeras로 분류하기
    def AutokerasLearn(self, X_train, X_test, y_train, y_test):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

        input_node = ak.Input()
        output_node = ak.RNNBlock(return_sequences=False)(input_node)
        output_node = ak.DenseBlock()(output_node)
        output_node = ak.ClassificationHead()(output_node)

        auto_model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            max_trials=10,
            overwrite=True,
            directory='../../data/model/autokeras_rnn',
        )

        history = auto_model.fit(
            X_train,
            y_train,
            epochs=20,
            validation_data=(X_val, y_val)
        )

        loaded_model = auto_model.export_model()
        tf.keras.models.save_model(loaded_model, '../../data/model/automlkeras_rnn_best_model.h5', save_format='h5')
        print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

        model_loss(history)

        return loaded_model

    # 리뷰 예측해보기

    def __init__(self):
        X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer = pretreatment()

        loaded_model_keras = self.AutokerasLearn(X_train, X_test, y_train, y_test)

        sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해", loaded_model_keras, stopwords, tokenizer, max_len)
        sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?", loaded_model_keras, stopwords, tokenizer, max_len)


autokeras = RNNAutoml()
