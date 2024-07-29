import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import ast
import pickle
import os.path
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import json
from sentence_transformers import SentencesDataset, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
import random

# Check if MPS is available and set the device accordingly
device = torch.device("mps")

# Load the dataset
def dataloader(data_filename):
    tweets = []
    #if tweets have been pickled, load the pickle file
    if os.path.isfile("tweets_" + data_filename + ".pkl"):
        with open ("tweets_" + data_filename + ".pkl", "rb") as f:
            tweets = pickle.load(f)
            return tweets
    #open the file, loop through every line, convert line to a dictionary, extract content
    with open("datasets/" + data_filename + ".txt", "r") as f:
        for line in f.readlines():
            try:
                tweet = ast.literal_eval(line.strip())
                tweet = [x for x in tweet["content"].split(" ") if "@" not in x and "http" not in x and "pic.twitter" not in x]
                tweets.append(" ".join(tweet))
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing line: {line.strip()}" + data_filename)
                print(f"Error message: {e}")
    with open("tweets_" + data_filename + ".pkl", "wb") as f:
        pickle.dump(tweets, f)
    return tweets

print("finished loading dataset")

data_filename = "evolution"
# Extract the sentences
sentences = dataloader(data_filename)

# Load pre-trained model configuration and model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to("mps")


# Load tweets and generate embeddings

def load_labeled_tweets(filenames, labels):
    sentences, y = [], []
    for filename, label in zip(filenames, labels):
        with open(filename, "rb") as f:
            if label == 0:
                tweets = pickle.load(f)[:96]
            elif label == 1:
                tweets = pickle.load(f)[:96]
            elif label == 2:
                tweets = pickle.load(f)[:96] 
            else:
                continue
            print(f"Label {label}: Loaded {len(tweets)} tweets")
            sentences.extend(tweets)
            y.extend([label] * len(tweets))
    return sentences, np.array(y)

def load_tweets():
    # File paths and labels
    filenames = [
        "tweets_test_data_evolution_true.pkl",
        "tweets_test_data_evolution_false.pkl",
        "tweets_test_data_not_evolution.pkl"
    ]
    labels = [0, 1, 2]

    dataloader("test_data_evolution_true")
    dataloader("test_data_evolution_false")
    dataloader("test_data_not_evolution")

    # Load labeled tweets labels
    labeled_sentences, y = load_labeled_tweets(filenames, labels)

    embeddings_filename = "embeddings_" + str(len(sentences)) + ".pkl"

    return labeled_sentences, y, embeddings_filename





def draw(labels, name, embeddings):
    plt.ion()

    # Plot the 2D PCA results with K-Means cluster labels
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    colors = plt.get_cmap('tab20', len(unique_labels))

    for label in unique_labels:
        class_member_mask = (labels == label)
        xy = embeddings[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=5, label=f'Cluster {label+1}', color=colors(label))

    plt.title(name)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()

    # Keep the plot displayed and allow the script to continue running
    plt.pause(0.001)

def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.pause(0.001)

# # Store metrics
# training_sizes = [40, 50, 60, 70, 80, 90]
# f1_scores = {0: [], 1: [], 2: []}
# precision_scores = {0: [], 1: [], 2: []}
# recall_scores = {0: [], 1: [], 2: []}

# # Train the SVM with varying training sizes
# for size in training_sizes:
#     # Subset training data
#     X_train, X_test, y_train, y_test = train_test_split(test_embeddings, y, train_size=size*3, stratify=y, random_state=42)

#     # Ensure that the subset includes at least one sample from each class
#     if len(np.unique(y_train)) < len(np.unique(y)):
#         print(f"Skipping training size {size} as it does not include all classes.")
#         continue

#     # Apply PCA to reduce dimensionality for better visualization
#     pca = PCA(n_components=min(40, test_embeddings.shape[1] - 1))
#     X_train_pca = pca.fit_transform(X_train)
#     full_embeddings_pca = pca.transform(full_embeddings)

#     # Apply t-SNE on the PCA-reduced data
#     tsne = TSNE(n_components=2, random_state=42)
#     X_train_tsne = tsne.fit_transform(X_train_pca)
#     full_embeddings_tsne = tsne.fit_transform(full_embeddings_pca)

#     # Train the SVM
#     svm = SVC(kernel='linear')
#     svm.fit(X_train_tsne, y_train)

#     # Predict and evaluate
#     y_pred = svm.predict(X_train_tsne)

#     # Confusion Matrix
#     cm = confusion_matrix(y_train, y_pred)
#     plot_confusion_matrix(cm, f'Confusion Matrix (Training Size: {size})')

#     # Calculate precision, recall, and f1 score for each class
#     for label in [0, 1, 2]:
#         precision_scores[label].append(precision_score(y_train, y_pred, labels=[label], average='micro'))
#         recall_scores[label].append(recall_score(y_train, y_pred, labels=[label], average='micro'))
#         f1_scores[label].append(f1_score(y_train, y_pred, labels=[label], average='micro'))

# # Plotting the metrics
# def plot_metric(metric_values, metric_name):
#     plt.figure(figsize=(10, 6))
#     for label, values in metric_values.items():
#         plt.plot(training_sizes, values, marker='o', label=f'Label {label}')
#     plt.title(f'{metric_name} vs Training Size')
#     plt.xlabel('Training Size')
#     plt.ylabel(metric_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    

# plot_metric(f1_scores, 'F1 Score')
# plot_metric(precision_scores, 'Precision')
# plot_metric(recall_scores, 'Recall')


def filter_data_for_svm(test_embeddings, y, test_sentences, evolution_claims): #expieriment2
    evolution_claims_set = set([claim[0] for claim in evolution_claims])
    filtered_embeddings = []
    filtered_labels = []
    filtered_sentences = []

    num_sentences_counter =0

    for embedding, label, sentence in zip(test_embeddings, y, test_sentences):
        if sentence in evolution_claims_set and label in [0, 1]:  # True/False labels only
            filtered_embeddings.append(embedding)
            filtered_labels.append(label)
            filtered_sentences.append(sentence)
            num_sentences_counter+=1
    print("Number of sentences: " + str(num_sentences_counter))
    return np.array(filtered_embeddings), np.array(filtered_labels), filtered_sentences


def calculate_cosine_similarity_accuracy(evolution_claims, test_sentences, y): #expieriment2
    evolution_claims_set = set([claim[0] for claim in evolution_claims])
    true_positives = 0
    total_evolution_claims = len(evolution_claims_set)

    for sentence, label in zip(test_sentences, y):
        if sentence in evolution_claims_set and label in [0, 1]:  # True/False labels only
            true_positives += 1

    accuracy = true_positives / total_evolution_claims if total_evolution_claims > 0 else 0
    return accuracy

def run_tsne(X_train, X_test, X_unrelated):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_unrelated = np.array(X_unrelated)
    
    tsne = TSNE(n_components=3, random_state=42, perplexity = 30, method = "exact")
    X_train_tsne = tsne.fit_transform(X_train)
    X_test_tsne = tsne.fit_transform(X_test)
    #X_unrelated_tsne = tsne.fit_transform(X_unrelated)

    return X_train_tsne, X_test_tsne



def make_report(X_train, y_train, X_test_evo_tsne, y_test_evo, y_test_combined,  y_train_pred_svm, y_test_pred_svm, y_test_pred_combined, labels):
    # print("Training Classification Report:")
    # print(classification_report(y_train, y_pred_train, target_names=labels))
    print("Testing Classification Report:")
    print(classification_report(y_test_combined, y_test_pred_combined, target_names=labels))

    # Confusion matrices
    # cm_train = confusion_matrix(y_train, y_pred_train)
    # plot_confusion_matrix(cm_train, 'Confusion Matrix - Training Set', labels)
    cm_test = confusion_matrix(y_test_combined, y_test_pred_combined)
    plot_confusion_matrix(cm_test, 'Confusion Matrix - Testing Set', labels)

    draw(y_train, "True Labels - Train Set", X_train)
    print("Finished drawing true labels for train set")
    # draw(y_test_evo, "True Labels - Test Set", X_test_evo_tsne)
    # print("Finished drawing true labels for test set")
    # draw(y_pred_train, "SVM Predicted Labels - Train Set", X_train)
    # print("Finished drawing SVM predictions for train set")
    # draw(y_test_pred_svm, "SVM Predicted Labels - Test Set", X_test_evo_tsne)
    # print("Finished drawing SVM predictions for test set")

    



def expieriment1(X_train, y_train, X_test, y_test):
    print("EXPIERIMENT 1:")
    X_train, X_test = run_tsne(X_train, X_test)
    labels = ["True", "False", "Unrelated"]
    y_pred_train, y_pred_test = run_svm(X_train, y_train, X_test, y_test, labels)
    make_report(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test)


 # Function to calculate cosine similarity
def calc_cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def filter_sentences_by_cos_sim(sentence_embeddings, sentence_labels, reference_embeddings, threshold = 0.4):
    X = []
    y = []
    X_unrealted = []
    y_unrelated_actual = []
    y_unrelated_pred = []


    # Identify sentences related to evolution based on cosine similarity
    for i, embedding in enumerate(sentence_embeddings):
        pass_threshold = False
        for ref_embedding in reference_embeddings:
            similarity = calc_cosine_similarity(embedding, ref_embedding)
            if similarity > threshold:
                X.append(embedding)
                y.append(sentence_labels[i])
                pass_threshold = True
                break
        if not pass_threshold:
            X_unrealted.append(embedding)
            y_unrelated_actual.append(sentence_labels[i])
            y_unrelated_pred.append(2)
                
    # Optional: Save the marked evolution claims to a file
    with open("evolution_claims.pkl", "wb") as f:
        pickle.dump(X, f)
    return X, y, X_unrealted, y_unrelated_actual, y_unrelated_pred

#filter training set (for SVM) to only include true, false (b/c unrelated is taken care of by cos_sims)
def only_evolution_claims(sentence_embeddings, sentence_labels):
    X_train, y_train = [], []
    for i, embedding in enumerate(sentence_embeddings):
        if sentence_labels[i] == 0 or sentence_labels[i] == 1:
            X_train.append(sentence_embeddings[i])
            y_train.append(sentence_labels[i])
    
    return X_train, y_train



def expieriment2(X_train, y_train, X_test, y_test):
    print("EXPIERIMENT 2:")
   
    # Define the list of reference claims about evolution
    reference_claims = [
        "A social media post about evolution",
        "Evolution is real",
        "A social media post about natural selection",
        "Natural selection is real",
        "God created humans",
        "We did not evolve from apes",
        "Adam and Eve were the first humans",
        "Evolution is fake"
    ]

    X_train_evo, y_train_evo = only_evolution_claims(X_train, y_train)

    # Get embeddings for the reference claims
    reference_embeddings = model.encode(reference_claims)

    reference_embeddings = np.concatenate((np.array(reference_embeddings), np.array(X_train_evo)))

    X_test_evo, y_test_evo, X_test_unrelated, y_test_unrelated_actual, y_test_unrelated_pred = filter_sentences_by_cos_sim(X_test, y_test, reference_embeddings, 0.7)

    X_train_evo_tsne, X_test_evo_tsne = run_tsne(X_train_evo, X_test_evo, X_test_unrelated)
    
    svm = SVC(kernel='rbf')
    svm.fit(X_train_evo, y_train_evo)

    y_train_pred_svm = svm.predict(X_train_evo)
    y_test_pred_svm = svm.predict(X_test_evo)

     # Combine unrelated predictions with SVM predictions
    y_test_pred_combined = np.concatenate((y_test_unrelated_pred, y_test_pred_svm))
    y_test_combined = np.concatenate((y_test_unrelated_actual, y_test_evo))
    #X_test_combined = np.concatenate((X_unrelated, X_test))

    labels = ["True", "False", "Unrealted"]
    
    make_report(X_train, y_train, X_test_evo_tsne, y_test_evo, y_test_combined, y_train_pred_svm, y_test_pred_svm, y_test_pred_combined, labels)

def expieriment2_tsne(X_train, y_train, X_test, y_test):
    print("EXPIERIMENT 2:")
   
    # Define the list of reference claims about evolution
    reference_claims = [
        "A social media post about evolution",
        "Evolution is real",
        "A social media post about natural selection",
        "Natural selection is real",
        "God created humans",
        "We did not evolve from apes",
        "Adam and Eve were the first humans",
        "Evolution is fake"
    ]

    X_train_evo, y_train_evo = only_evolution_claims(X_train, y_train)

    # Get embeddings for the reference claims
    reference_embeddings = model.encode(reference_claims)

    reference_embeddings = np.concatenate((np.array(reference_embeddings), np.array(X_train_evo)))

    X_test_evo, y_test_evo, X_test_unrelated, y_test_unrelated_actual, y_test_unrelated_pred = filter_sentences_by_cos_sim(X_test, y_test, reference_embeddings, 0.7)

    X_train_evo_tsne, X_test_evo_tsne = run_tsne(X_train_evo, X_test_evo, X_test_unrelated)
    
    svm = SVC(kernel='rbf')
    svm.fit(X_train_evo_tsne, y_train_evo)

    y_train_pred_svm = svm.predict(X_train_evo_tsne)
    y_test_pred_svm = svm.predict(X_test_evo_tsne)

     # Combine unrelated predictions with SVM predictions
    y_test_pred_combined = np.concatenate((y_test_unrelated_pred, y_test_pred_svm))
    y_test_combined = np.concatenate((y_test_unrelated_actual, y_test_evo))
    #X_test_combined = np.concatenate((X_unrelated, X_test))

    labels = ["True", "False", "Unrealted"]
    
    make_report(X_train, y_train, X_test_evo_tsne, y_test_evo, y_test_combined, y_train_pred_svm, y_test_pred_svm, y_test_pred_combined, labels)


def expieriment3(X_train, y_train, X_test, y_test):
    print("EXPIERIMENT 2:")
   
    # Define the list of reference claims about evolution
    reference_claims = [
        "A social media post about evolution",
        "Evolution is real",
        "A social media post about natural selection",
        "Natural selection is real",
        "God created humans",
        "We did not evolve from apes",
        "Adam and Eve were the first humans",
        "Evolution is fake"
    ]

    X_train_evo, y_train_evo = only_evolution_claims(X_train, y_train)

    # Get embeddings for the reference claims
    reference_embeddings = model.encode(reference_claims)

    reference_embeddings = np.concatenate((np.array(reference_embeddings), np.array(X_train_evo)))

    X_test_evo, y_test_evo, X_test_unrelated, y_test_unrelated_actual, y_test_unrelated_pred = filter_sentences_by_cos_sim(X_test, y_test, reference_embeddings, 0.7)


    y_test_pred = []
    for i, post in enumerate(X_test_evo):
        max_cos_sim = 0
        label_of_max = None
        for j, ref_post in enumerate(X_train_evo):
            similarity = calc_cosine_similarity(post, ref_post)
            if max_cos_sim < similarity:
                max_cos_sim = similarity
                label_of_max = y_train_evo[j]
        y_test_pred.append(label_of_max)


     # Combine unrelated predictions with SVM predictions
    y_test_pred_combined = np.concatenate((y_test_unrelated_pred, y_test_pred))
    y_test_combined = np.concatenate((y_test_unrelated_actual, y_test_evo))
    #X_test_combined = np.concatenate((X_unrelated, X_test))

    labels = ["True", "False", "Unrealted"]
    
    make_report(X_train, y_train, X_test_evo, y_test_evo, y_test_combined, None, y_test_pred, y_test_pred_combined, labels)







if __name__ == "__main__":
    labeled_sentences, y, embeddings_filename = load_tweets()

    # Get embeddings
    if os.path.isfile(embeddings_filename):
        with open (embeddings_filename, "rb") as f:
            full_embeddings = pickle.load(f)
    else:
        full_embeddings = model.encode(sentences)
        with open(embeddings_filename, "wb") as f:
            pickle.dump(full_embeddings, f)

    print("finished with embeddings")

    num_sentences = len(sentences)

    X = np.array(model.encode(labeled_sentences))


    X_train, X_test, y_train, y_test, sentences_train, sentences_test = train_test_split(X, y, labeled_sentences, test_size=0.25, random_state=42, stratify=y)

    


    #expieriment1()
    #expieriment2(X_train, y_train, X_test, y_test)
    #expieriment2_tsne(X_train, y_train, X_test, y_test)
    #expieriment3(X_train, y_train, X_test, y_test)



    input("Press [enter] to end.")