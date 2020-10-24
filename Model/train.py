import json
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"


LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32

NUM_KEYWORDS = 10



def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    #extract inputs and targets
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training data is loaded!")

    return X, y

def get_data_split(data_path, test_size = 0.2, test_validation = 0.2):

    #load the dataset
    X, y =  load_dataset(data_path)

    #create train/validation test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    #convert inputs from 2-D to 3-D arrays
    #(#segments, 13[MFCCs], 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    y_validation = y_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def build_model(input_shape, learning_rate, error = "sparse_categorical_crossentropy"):

    #build network
    model = keras.Sequential()

    #conv layer 1
    model.add(keras.layers.Conv2D(64, (3,3), activation = "relu",
                                  input_shape = input_shape,
                                  kernel_regularizer = keras.regularizers.l2(0.001) ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides = (2,2), padding = "same"))
    #conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    #conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    #flatten the output and feed it into a dense layer(pooling)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = "relu"))  #fully connected layer
    model.add(keras.layers.Dropout(0.3))   #it shuts down 30% of the neurons in the dense layer (done stoichastically)  [Made to adapt and all diff neurons take part and equal responsibiluity]

    #softmax classifier in the end
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation = "softmax" ))  # output is an array #for 3 keywords [0.1,0.7,0.2] value and score

    #compile the model(using keras, tensorflow)
    optimiser = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer = optimiser, loss = error, metrics = ["accuracy"])

    #print model overview
    model.summary()
    return model

def main():
    #load the train validation and test data splits
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_split(DATA_PATH)


    #build CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) #( #segments, no of co-eff = 13(MFCC), 1 )
    model = build_model(input_shape, learning_rate= LEARNING_RATE)

    #train the model

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))
    #evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Error:{test_error}, Test Accuracy:{test_accuracy}")

    #save the model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
