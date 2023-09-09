
from sklearn.metrics import recall_score, roc_auc_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix



def train_svm(features, labels):
    X_train, X_test, y_train, y_test = split_data(features, labels)
    svm_model = create_and_train_svm(X_train, y_train)
    save_svm_model(svm_model)
    evaluate_svm_model(svm_model, X_test, y_test)

def split_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def create_and_train_svm(X_train, y_train):
    svm = SVC(kernel='linear', C=10, gamma=10, probability=True)
    svm.fit(X_train, y_train)
    return svm

def save_svm_model(model):
    with open('Thesis/svm_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def evaluate_svm_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)
    
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall:", recall)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-score:", f1)
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:", cm)
  
    y_pred_decision = model.decision_function(X_test)
    auc = roc_auc_score(y_test, y_pred_decision)
    print("AUC:", auc)
