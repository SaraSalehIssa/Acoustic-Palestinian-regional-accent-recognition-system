import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Function to load audio files and their corresponding labels
def load_audio_files(directory):
    audio_files = []
    labels = []
    sample_rates = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                try:
                    audio, sr = librosa.load(path, sr=None)
                    audio_files.append(audio)
                    labels.append(label)
                    sample_rates.append(sr)
                except Exception as e:
                    print(f"Error loading {file} from {label}: {e}!")
    return audio_files, labels, sample_rates

# Function to plot the distribution of audio file labels
def plot_data_distribution(labels, title, color='skyblue'):
    label_counts = Counter(labels)
    if not label_counts:
        print(f"No labels found for {title}. Please check your data directory and files.")
        return
    labels, counts = zip(*label_counts.items())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=color)
    plt.xlabel('Labels')
    plt.ylabel('Number of audio files')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# Function to extract MFCC features from audio
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=10)
    feature_vector = np.hstack((mfccs.mean(axis=1)))
    return feature_vector

# Function to train and evaluate various models
def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    gmm_models = {label: GaussianMixture(n_components=1, covariance_type='diag', random_state=42) for label in np.unique(y_train)}
    for label in gmm_models:
        gmm_models[label].fit(X_train[y_train == label])

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    models = {
        "SVM": svm_model,
        "KNN": knn_model,
        "GMM": gmm_models,
        "Random Forest": rf_model
    }
    accuracies = {}

    for name, model in models.items():
        if name == "GMM":
            y_pred = []
            for x in X_val:
                scores = {label: model[label].score([x]) for label in model}
                y_pred.append(max(scores, key=scores.get))
            y_pred = np.array(y_pred)
        else:
            y_pred = model.predict(X_val)
        accuracies[name] = accuracy_score(y_val, y_pred)
        print(f"{name} Accuracy: {accuracies[name]:.2f}")

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple']
    plt.bar(accuracies.keys(), accuracies.values(), color=colors)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.show()

    return models

# Function to evaluate models on the test set with additional metrics
def evaluate_on_test_set_with_metrics(models, X_test, y_test):
    metrics = {}
    for name, model in models.items():
        if name == "GMM":
            y_pred = []
            for x in X_test:
                scores = {label: model[label].score([x]) for label in model}
                y_pred.append(max(scores, key=scores.get))
            y_pred = np.array(y_pred)
        else:
            y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        metrics[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        cm = confusion_matrix(y_test, y_pred)
        accuracy_from_cm = np.trace(cm) / np.sum(cm)
        print(f"{name} Test Accuracy: {accuracy:.2f}")
        print(f"{name} Test Accuracy from Confusion Matrix: {accuracy_from_cm:.2f}")
        print(f"{name} Test Precision: {precision:.2f}")
        print(f"{name} Test Recall: {recall:.2f}")
        print(f"{name} Test F1 Score: {f1:.2f}")
        print("---------------------------------------------------------------------------------")

    # Plotting the metrics
    plot_metrics(metrics)

# Function to plot metrics comparison for different models
def plot_metrics(metrics):
    labels = metrics.keys()
    accuracy = [metrics[label]['accuracy'] for label in labels]
    precision = [metrics[label]['precision'] for label in labels]
    recall = [metrics[label]['recall'] for label in labels]
    f1_score = [metrics[label]['f1_score'] for label in labels]

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, accuracy, width, label='Accuracy')
    plt.bar(x, precision, width, label='Precision')
    plt.bar(x + width, recall, width, label='Recall')
    plt.bar(x + 2 * width, f1_score, width, label='F1 Score')

    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Metrics Comparison')
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.legend()

    for i in range(len(labels)):
        plt.text(i - width, accuracy[i] + 0.01, f"{accuracy[i]:.2f}", ha='center')
        plt.text(i, precision[i] + 0.01, f"{precision[i]:.2f}", ha='center')
        plt.text(i + width, recall[i] + 0.01, f"{recall[i]:.2f}", ha='center')
        plt.text(i + 2 * width, f1_score[i] + 0.01, f"{f1_score[i]:.2f}", ha='center')

    plt.show()

# Function to predict the country from a single audio file
def predict_country(audio_path, model, label_encoder, scaler, pca=None, is_gmm=False):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        features = extract_features(audio, sr)
        features = scaler.transform([features])
        if pca:
            features = pca.transform(features)
        if is_gmm:
            scores = {label: model[label].score(features) for label in model}
            prediction = max(scores, key=scores.get)
            country = label_encoder.inverse_transform([prediction])
        else:
            prediction = model.predict(features)
            country = label_encoder.inverse_transform(prediction)
        return country[0]
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Directories for training and testing data
train_dir = 'C:/Users/hp/Desktop/Spoken Project/training data'
test_dir = 'C:/Users/hp/Desktop/Spoken Project/testing data'

# Load and plot data
train_audio, train_labels, train_sr = load_audio_files(train_dir)
test_audio, test_labels, test_sr = load_audio_files(test_dir)

plot_data_distribution(train_labels, 'Training Data Distribution', color='lightcoral')
plot_data_distribution(test_labels, 'Testing Data Distribution', color='lightseagreen')

# Extract features from audio
train_features = np.array([extract_features(audio, sr) for audio, sr in zip(train_audio, train_sr)])
test_features = np.array([extract_features(audio, sr) for audio, sr in zip(test_audio, test_sr)])

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Encode the labels
label_encoder = LabelEncoder()
label_encoder.fit(train_labels + test_labels)

y_train_encoded = label_encoder.transform(train_labels)
y_test_encoded = label_encoder.transform(test_labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, y_train_encoded, test_size=0.2, random_state=42)

# Train and evaluate models
print("---------------------------------------------------------------------------------")
print("Training data")
models = train_and_evaluate_models(X_train, y_train, X_val, y_val)
print("---------------------------------------------------------------------------------")

# Evaluate on the test set
print("Testing data")
evaluate_on_test_set_with_metrics(models, test_features, y_test_encoded)
print("---------------------------------------------------------------------------------")

# Predict single audio file for all models
audio_path = 'C:/Users/A To Z/PycharmProjects/spoken2/TestingData/testing data/Nablus/nablus_train046.wav'
for model_name, model in models.items():
    is_gmm = model_name == "GMM"
    predicted_country = predict_country(audio_path, model, label_encoder, scaler, is_gmm=is_gmm)
    print(f"The predicted country for the audio file using {model_name} is: {predicted_country}")
