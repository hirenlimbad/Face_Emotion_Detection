import components.data_ingetion as data_ingetion
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import SGD


class model_training:
    def __init__(self):
        datai = data_ingetion.DataIngetion()
        datai.collect_data(r'/home/hiren/Desktop/Vision Beyond/ML_end_to_end/artifacts/images_data/images_data/ferdata/train')
        self.X, self.y = datai.get_data()

    def data_preprocessing(self):
        self.X = np.array(self.X) / 255.0
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state=78)
        X_val, X_train = X_train[:1000], X_train[1000:]
        y_val, y_train = y_train[:1000], y_train[1000:]

        x_train = np.array(X_train).reshape(-1, 48, 48, 1)
        x_test = np.array(X_test).reshape(-1, 48, 48, 1)
        x_val = np.array(X_val).reshape(-1, 48, 48, 1)

        # Create a dataset from your NumPy arrays
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # Define batch size and shuffle buffer size
        batch_size = 12
        shuffle_buffer_size = len(x_train)

        # Shuffle and batch the datasets
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        return train_dataset, validation_dataset, test_dataset

    def train(self):

        train_dataset, validation_dataset, test_dataset = self.data_preprocessing()

        cnn = Sequential([
            layers.Input(shape=(48, 48, 1)),

            layers.Conv2D(
                filters=64,
                kernel_size=(5,5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_1'
            ),
            layers.BatchNormalization(name='batchnorm_1'),
            layers.Conv2D(
                filters=64,
                kernel_size=(5,5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_2'
            ),
            layers.BatchNormalization(name='batchnorm_2'),
            layers.MaxPooling2D(pool_size=(2,2), name='maxpooling1'),
            layers.Dropout(0.4, name='dropout_1'),
            layers.Conv2D(
                filters=128,
                kernel_size=(5,5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_3'
            ),
            layers.BatchNormalization(name='batchnorm_3'),
            layers.Conv2D(
                filters=128,
                kernel_size=(5,5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_4'
            ),
            layers.BatchNormalization(name='batchnorm_4'),
            layers.MaxPooling2D(pool_size=(2,2), name='maxpooling2'),
            layers.Dropout(0.4, name='dropout_2'),

            layers.Conv2D(
                filters=128,
                kernel_size=(5,5),
                activation='elu',
                padding='same',
                kernel_initializer='he_normal',
                name='conv2d_5'
            ),
            layers.BatchNormalization(name='batchnorm_5'),
            layers.Dropout(0.4, name='dropout_3'),
        ])

        # Create the final model
        model = Sequential([
            cnn,
            layers.Flatten(name='flatten'),
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.Dense(7, activation='softmax', name='output_layer')
        ])
        model.build(input_shape=(None, 48, 48, 1))
        model.summary()

        # Compile the model
        optimizer = SGD(learning_rate=0.01)
        model.compile(
            optimizer=optimizer,
            metrics=['accuracy'],
            loss="sparse_categorical_crossentropy"
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model.fit(train_dataset, epochs=1, validation_data=validation_dataset, callbacks=[early_stopping])

        self.model_score = model.evaluate(test_dataset)
        self.model = model
        self.model.save("ML_end_to_end/artifacts/saved_model/")

if __name__ == "__main__":
    model_training = model_training()
    model_training.train()
    print("Model score",model_training.model_score)