from keras.layers import Dense, Flatten, Dropout, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras_models.text_classifier import KerasTextClassifier


class FCholletCNN(KerasTextClassifier):
    """Based on
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    except with trainable embeddings instead of pretrained from GloVe"""

    def __init__(
            self,
            embedding_dim=30,
            embeddings_path=None,
            optimizer='adam',
            batch_size=32,
            epochs=10,
            units=128,
            dropout_rate=0.25,
            vocab_size=None,
            vocab=None):
        super(FCholletCNN, self).__init__(
            embedding_dim=embedding_dim,
            embeddings_path=embeddings_path,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            vocab=vocab,
            vocab_size=vocab_size)

        self.units = units
        self.dropout_rate = dropout_rate
        self.params['units'] = units
        self.params['dropout_rate'] = dropout_rate

    def transform_embedded_sequences(self, embedded_sequences):
        x = Conv1D(self.units, 5, activation='relu', name='c1')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(self.units, 5, activation='relu', name='c2')(x)
        x = MaxPooling1D(5)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(self.units, 5, activation='relu', name='c3', data_format='channels_first')(x)
        x = MaxPooling1D(35)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = GlobalMaxPooling1D()(x)
        # x = Flatten()(x)
        x = Dense(self.units, activation='relu')(x)
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds
