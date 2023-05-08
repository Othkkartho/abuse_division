from scr.training.pretreatment import pretreatment, sentiment_predict, model_loss
import autokeras as ak
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class CNNAutoml:
    def Multiautokeras(self, vocab_size, X_train, X_test, y_train, y_test):
        embedding_dim = 128
        dropout_ratio = (0.5, 0.8)
        num_filters = 128
        hidden_units = 128

        # define the input shape
        input_node = ak.Input()

        # add an embedding layer
        output_node = ak.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            dropout=dropout_ratio[0]
        )(input_node)

        # add convolutional layers with different kernel sizes
        for sz in [3, 4, 5]:
            conv = ak.ConvBlock(
                filters=num_filters,
                kernel_size=sz,
                dropout=dropout_ratio[0],
                pooling='global_max'
            )(output_node)
            output_node = ak.Merge()([output_node, conv])

        # add a dense layer
        output_node = ak.DenseBlock(
            units=hidden_units,
            dropout=dropout_ratio[1],
            activation='relu'
        )(output_node)

        # add the final output layer
        output_node = ak.DenseBlock(
            units=1,
            activation='sigmoid'
        )(output_node)

        # define the autokeras model
        model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=10
        )

        # compile the model
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["acc"]
        )

        # fit the autokeras model to the training data
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('../../data/model/CNN_mk_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        history = model.fit(
            X_train,
            y_train,
            batch_size=64,
            epochs=10,
            validation_split=0.2,
            verbose=2,
            callbacks=[es, mc]
        )

        # load the best model found by autokeras
        loaded_model = ak.models.load_model('../../data/model/CNN_automd_model', verbose=1)

        model_loss(history)

        # evaluate the model on the test data
        print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

        return loaded_model

    def oneautoml(self, X_train, X_test, y_train, y_test):
        embedding_dim = 128
        dropout_ratio = (0.5, 0.8)
        num_filters = 128
        hidden_units = 128

        # Define the input shape
        input_node = ak.Input()

        # Add the embedding layer
        z = ak.layers.Embedding()(input_node)
        z = ak.layers.Dropout(dropout_ratio[0])(z)

        conv_blocks = []

        for sz in [3, 4, 5]:
            # Add the convolutional layer
            conv = ak.layers.ConvBlock(
                kernel_size=sz,
                num_blocks=1,
                num_layers=1,
                num_filters=num_filters,
                dropout=dropout_ratio[1],
                max_pooling=True,
                activation="relu",
            )(z)
            conv_blocks.append(conv)

        z = ak.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        z = ak.layers.DenseBlock(units=hidden_units, dropout=dropout_ratio[1], activation="relu")(z)

        # Add the output layer
        output_node = ak.layers.Dense(units=1, activation="sigmoid")(z)

        # Build the AutoKeras model
        model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=10)

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

        # Evaluate the model
        score = model.evaluate(X_test, y_test)
        print("Test accuracy:", score)

        # Get the best model found by AutoKeras
        best_model = model.export_model()

        # Save the model
        best_model.save("../../data/model/CNN_autood_model.h5")

        model_loss(history)

        return best_model

    def __init__(self):
        X_train, X_test, y_train, y_test, vocab_size, max_len, stopwords, tokenizer = pretreatment()

        loaded_model_automd = self.Multiautokeras(vocab_size, X_train, X_test, y_train, y_test)
        loaded_model_autood = self.oneautoml(vocab_size, X_train, X_test, y_train, y_test)

        sentiment_predict("미친 새끼 또 저 지랄이네 대단하다 대단해", loaded_model_automd, stopwords, tokenizer, max_len)
        sentiment_predict("안녕하세요 오랜만에 뵈요. 어떤일 하고 있나요?", loaded_model_automd, stopwords, tokenizer, max_len)
