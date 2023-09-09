
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import pickle 


def split_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=120, min_samples_leaf=10, random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    return rf_model

def save_rf_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def evaluate_rf_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    predicted_labels = ['Benign' if pred == 0 else 'Malignant' for pred in y_pred]
    
    accuracy = accuracy_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels, pos_label='Malignant')
    precision = precision_score(y_test, predicted_labels, pos_label='Malignant')
    f1 = f1_score(y_test, predicted_labels, pos_label='Malignant')
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("AUC:", auc)
    print("Confusion Matrix:", cm)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1-score:", f1)
    

def train_and_evaluate_rf(features, labels):
    X_train, X_test, y_train, y_test = split_data(features, labels)
    rf_model = train_random_forest(X_train, y_train)
    save_rf_model(rf_model, 'Thesis/rf_model.pkl')
    evaluate_rf_model(rf_model, X_test, y_test)


