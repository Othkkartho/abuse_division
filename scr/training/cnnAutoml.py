from scr.training.pretreatment import pretreatment, sentiment_predict, model_loss
import autokeras as ak
import numpy as np
import tensorflow as tf


class CNNAutoml:
    def Multiautokeras(self, X_train, X_test, y_train, y_test):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        input_node = ak.Input()
        output_node = ak.ConvBlock()(input_node)
        output_node = ak.DenseBlock()(output_node)
        output_node = ak.ClassificationHead()(output_node)

        auto_model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            max_trials=10,
            overwrite=True,
            directory='../../data/model/autokeras_cnn',
        )

        history = auto_model.fit(X_train, y_train, epochs=20, validation_split=0.2)
        loaded_model = auto_model.export_model()
        tf.keras.models.save_model(loaded_model, '../../data/model/automlkeras_cnn_best_model.h5', save_format='h5')
        print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

        model_loss(history)

        return loaded_model

    def __init__(self):
        X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer = pretreatment()

        loaded_model_automd = self.Multiautokeras(vocab_size, X_train, X_test, y_train, y_test, max_len)

        sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해", loaded_model_automd, stopwords, tokenizer, max_len)
        sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?", loaded_model_automd, stopwords, tokenizer, max_len)


CNNAutoml()
