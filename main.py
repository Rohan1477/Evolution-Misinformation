import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import ast
import pickle
import os.path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
from sklearn.model_selection import train_test_split
import random

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
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
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
                tweets = pickle.load(f)[:100]
            elif label == 1:
                tweets = pickle.load(f)[:100]
            elif label == 2:
                tweets = pickle.load(f)[:100] 
            else:
                continue
            print(f"Label {label}: Loaded {len(tweets)} tweets")
            sentences.extend(tweets)
            y.extend([label] * len(tweets))
    return sentences, np.array(y)

def load_tweets():
    filenames = [
        "tweets_test_data_evolution_true.pkl",
        "tweets_test_data_evolution_false.pkl",
        "tweets_test_data_not_evolution.pkl"
    ]

    labels = [0, 1, 2]

    dataloader("test_data_evolution_true")
    dataloader("test_data_evolution_false")
    dataloader("test_data_not_evolution")

    labeled_sentences, y = load_labeled_tweets(filenames, labels)
    embeddings_filename = "embeddings_" + str(len(sentences)) + ".pkl"

    return labeled_sentences, y, embeddings_filename

def draw(labels, name, embeddings):
    plt.ion()
    label_map = {2: 'Unrelated', 0: 'True', 1: 'False'}
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    colors = plt.get_cmap('tab20', len(unique_labels))
    label_order = [2, 0, 1]
    for label in label_order:
        class_member_mask = (labels == label)
        xy = embeddings[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=5, label=label_map[label], color=colors(label))
    plt.title(name)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()
    plt.pause(0.001)

def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.pause(0.001)

def make_report(exp_name, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, no_y_train_pred = False):
    labels = ["True", "False", "Unrelated"]
    if not no_y_train_pred:
        print(exp_name + " Training Classification Report:")
        print(classification_report(y_train, y_train_pred, target_names=labels, zero_division = 0))
    try:
        print(exp_name + " Testing Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=labels, zero_division = 0))
    except: pass
    pred_counts = Counter(y_test_pred)
    pred_counts_df = pd.DataFrame(pred_counts.items(), columns=['Class', 'Count'])
    print(pred_counts_df)

    # Confusion matrices
    if not no_y_train_pred:
        cm_train = confusion_matrix(y_train, y_train_pred)
        plot_confusion_matrix(cm_train, exp_name + ' Confusion Matrix - Training Set', labels)
    try:
        cm_test = confusion_matrix(y_test, y_test_pred)
        plot_confusion_matrix(cm_test, exp_name + ' Confusion Matrix - Testing Set', labels)
    except: pass

    try: draw(y_train, exp_name + " Train Set - True Labels", X_train)
    except: pass
    if not no_y_train_pred: draw(y_train_pred, exp_name + " Train Set - Predicted Labels", X_train)
    try: draw(y_test, exp_name + " Test Set - True Labels", X_test)
    except: pass
    try: draw(y_test_pred, exp_name + " Test Set - Predicted Labels", X_test)
    except: pass

def generateSampleResults(y_test, y_test_pred, label_wanted):
    true, false, unrelated= [], [], []
    for i, (yt, yp) in enumerate(zip(y_test, y_test_pred)):
        if yp == 0:
            true.append(sentences_test[i])
        if yp == 1:
            false.append(sentences_test[i])
        if yp == 2:
            unrelated.append(sentences_test[i])

    with open("sample_results.txt", "w") as f:
        f.write("TRUE: \n")
        for post in random.sample(true, 30):
            try:
                f.write(str(post) + "\n")
            except: pass
        f.write("\nFALSE: \n")
        for post in random.sample(false, 30):
            try:
                f.write(str(post) + "\n")
            except: pass
        f.write("\nUNRELATED: \n")
        for post in random.sample(unrelated, 30):
            try:
                f.write(str(post) + "\n")
            except: pass

    # tp, fp, tn, fn = [], [], [], []
    # for i, (yt, yp) in enumerate(zip(y_test, y_test_pred)):
    #     if yt == label_wanted and yp == label_wanted:
    #         tp.append(sentences_test[i])
    #     if yt != label_wanted and yp == label_wanted:
    #         fp.append(sentences_test[i])
    #     if yt != label_wanted and yp != label_wanted:
    #         tn.append(sentences_test[i])
    #     if yt == label_wanted and yp != label_wanted:
    #         fn.append(sentences_test[i])

    # with open("sample_results.txt", "w") as f:
    #     f.write("TRUE POSITIVES: \n")
    #     for post in tp[:30]:
    #         f.write(str(post) + "\n")
    #     f.write("\nFALSE POSITIVES: \n")
    #     for post in fp[:30]:
    #             f.write(str(post) + "\n")
    #     f.write("\nTRUE NEGATIVES: \n")
    #     for post in tn[:30]:
    #         f.write(str(post) + "\n")
    #     f.write("\nFALSE NEGATIVES: \n")
    #     for post in fn[:30]:
    #         f.write(str(post) + "\n")

def calc_cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def filter_sentences_by_cos_sim(sentence_embeddings, sentence_labels, reference_embeddings, threshold = 0.67):
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
    #with open("evolution_claims.pkl", "wb") as f:
     #   pickle.dump(X, f)
    return X, y, X_unrealted, y_unrelated_actual, y_unrelated_pred

#filter training set (for SVM) to only include true, false (b/c unrelated is taken care of by cos_sims)
def only_evolution_claims(sentence_embeddings, sentence_labels):
    X_train, y_train = [], []
    for i, embedding in enumerate(sentence_embeddings):
        if sentence_labels[i] == 0 or sentence_labels[i] == 1:
            X_train.append(sentence_embeddings[i])
            y_train.append(sentence_labels[i])
    
    return X_train, y_train

def experiment1(X_train, y_train, X_test, y_test):
    exp_name = "Exp 1:"
    print("EXPERIMENT 1:")

    pca = PCA(n_components=min(23, np.concatenate((np.array(X_train), np.array(X_test))).shape[1] - 1))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC(kernel='rbf')
    svm.fit(X_train_pca, y_train)

    y_train_pred_svm = svm.predict(X_train_pca)
    y_test_pred_svm = svm.predict(X_test_pca)
    
    make_report(exp_name, X_train, y_train, X_test_pca, y_test, y_train_pred_svm, y_test_pred_svm)
    generateSampleResults(y_test, y_test_pred_svm, 1)

def experiment2(X_train, y_train, X_test, y_test):
    exp_name = "Exp 2:"
    print("EXPERIMENT 2:")

    X_train_evo, y_train_evo = only_evolution_claims(X_train, y_train)
    X_test_evo, y_test_evo, X_test_unrelated, y_test_unrelated_actual, y_test_unrelated_pred = filter_sentences_by_cos_sim(X_test, y_test, np.array(X_train_evo), 0.63)
    
    svm = SVC(kernel='rbf')
    svm.fit(X_train_evo, y_train_evo)

    y_train_pred_svm = svm.predict(X_train_evo)
    y_test_pred_svm = svm.predict(X_test_evo)

    # Combine unrelated predictions with SVM predictions
    y_test_pred_combined = np.concatenate((y_test_unrelated_pred, y_test_pred_svm))
    y_test_combined = np.concatenate((y_test_unrelated_actual, y_test_evo))

    #can't generate report for train vs train_pred b/c the training set itsn't filteted to take out the evolution claims using cos sim b/c the cos sim is comparing against the training set itself to determine if evo or not evo. Instead filtering is done perfectly (only_evolution_claims) (this is necesasry to train the SVM)
    make_report(exp_name, X_train, y_train, X_test, y_test_combined, y_train_pred_svm, y_test_pred_combined, no_y_train_pred = True)

def experiment2_pca(X_train, y_train, X_test, y_test):
    exp_name = "Exp 2_PCA:"
    print("EXPERIMENT 2_PCA:")

    X_train_evo, y_train_evo = only_evolution_claims(X_train, y_train)
    X_test_evo, y_test_evo, X_test_unrelated, y_test_unrelated_actual, y_test_unrelated_pred = filter_sentences_by_cos_sim(X_test, y_test, np.array(X_train_evo), 0.63)

    pca = PCA(n_components=min(54, X.shape[1] - 1))
    X_train_evo_pca = pca.fit_transform(X_train_evo)
    X_test_evo_pca = pca.transform(X_test_evo)
    
    svm = SVC(kernel='rbf')
    svm.fit(X_train_evo_pca, y_train_evo)

    y_train_pred_svm = svm.predict(X_train_evo_pca)
    y_test_pred_svm = svm.predict(X_test_evo_pca)

    # Combine unrelated predictions with SVM predictions
    y_test_pred_combined = np.concatenate((y_test_unrelated_pred, y_test_pred_svm))
    y_test_combined = np.concatenate((y_test_unrelated_actual, y_test_evo))

    make_report(exp_name, X_train, y_train, X_test, y_test_combined, y_train_pred_svm, y_test_pred_combined, no_y_train_pred = True)

def experiment3(X_train, y_train, X_test, y_test):
    exp_name = "Exp 3:"
    print("EXPERIMENT 3:")
 
    X_train_evo, y_train_evo = only_evolution_claims(X_train, y_train)
    X_test_evo, y_test_evo, X_test_unrelated, y_test_unrelated_actual, y_test_unrelated_pred = filter_sentences_by_cos_sim(X_test, y_test, np.array(X_train_evo), 0.63)

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

    make_report(exp_name, X_train, y_train, X_test, y_test_combined, y_train, y_test_pred_combined, no_y_train_pred = True)

if __name__ == "__main__":
    use_full_dataset = True

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

    X = np.array(model.encode(labeled_sentences))

    X_train, X_test, y_train, y_test, sentences_train, sentences_test = train_test_split(X, y, labeled_sentences, test_size=0.20, random_state=42, stratify=y)

    if use_full_dataset:
        X_test = full_embeddings
        y_test = np.zeros(len(X_test))
        sentences_test = sentences

    experiment1(X_train, y_train, X_test, y_test)
    experiment2(X_train, y_train, X_test, y_test)
    experiment2_pca(X_train, y_train, X_test, y_test)
    experiment3(X_train, y_train, X_test, y_test)

    input("Press [enter] to end.") 