import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import recall_score, roc_auc_score, f1_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pickle 


def load_data(data_dir, classes):
    data = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 400), interpolation=cv2.INTER_AREA)
            img = img / 255.0
            data.append(img)
            labels.append(classes.index(cls))
    return np.array(data), np.array(labels)

def preprocess_data(data, labels):
    data = np.expand_dims(data, axis=-1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, X_test, y_train, y_test

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def train_cnn_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    epochs = 15
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])

def save_cnn_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def evaluate_cnn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    recall = recall_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes)
    
    auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    print("AUC:", auc)
    print("Confusion Matrix:", cm)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1-score:", f1)
    
    accuracy_test = model.evaluate(X_test, y_test)
    print("Test accuracy:", accuracy_test)

def train_and_evaluate_cnn():
    data_dir = "Thesis/CBIS"
    classes = ["Benign", "Malignant"]
    data, labels = load_data(data_dir, classes)
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)
    input_shape = X_train[0].shape
    model = build_cnn_model(input_shape)
    trained_model = train_cnn_model(model, X_train, y_train, X_test, y_test)
    save_cnn_model(trained_model, 'Thesis/cnn_model.pkl')
    evaluate_cnn_model(model, X_test, y_test)


