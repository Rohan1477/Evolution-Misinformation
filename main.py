import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import ast
import pickle
import os.path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
def dataloader():
    tweets = []
    #if tweets have been pickled, load the pickle file
    if os.path.isfile("tweets.pkl"):
        with open ("tweets.pkl", "rb") as f:
            tweets = pickle.load(f)
            return tweets
    #open the file, loop through every line, convert line to a dictionary, extract content
    with open("datasets/evolution.txt", "r") as f:
        for line in f.readlines():
            tweet = ast.literal_eval(line)
            tweets.append(tweet['content'])
    with open("tweets.pkl", "wb") as f:
        pickle.dump(tweets, f)
    return tweets

print("finished loading dataset")

# Extract the sentences
sentences = dataloader()

# Load pre-trained model configuration and model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')



# Get embeddings
embeddings_filename = "embeddings_" + str(len(sentences)) + ".pkl"

if os.path.isfile(embeddings_filename):
    with open (embeddings_filename, "rb") as f:
        embeddings = pickle.load(f)
else:
    embeddings = model.encode(sentences)
    with open(embeddings_filename, "wb") as f:
        pickle.dump(embeddings, f)



print("finished with embeddings")
# Calculate cosine similarity between each pair of embeddings
def calc_cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# Compute cosine similarity matrix
num_sentences = len(sentences)



cosine_similarity_filename = "cosine_similarity_" + str(len(sentences)) + ".pkl"

# def generate_cosine_similarities(batch_size=3000):
#     cosine_similarities = {}
#     start_index = 0

#     if os.path.isfile(cosine_similarity_filename):
#         print("load file")
#         with open (cosine_similarity_filename, "rb") as f:
#             cosine_similarities = pickle.load(f)
#             # Determine the last completed index
#             print("Determine the last completed index")
#             keys = list(cosine_similarities.keys())
#             last_key = keys[-1] if keys else None
#             if last_key:
#                 start_index = max(int(last_key.split('/')[0]), int(last_key.split('/')[1])) + 1
#             print(f"Resuming from index: {start_index}")
#     for i in range(start_index, num_sentences, batch_size):
#         for j in range(i, min(i + batch_size, num_sentences)):
#             print(j)
#             for k in range(j):
#                 cosine_similarities[sentences[j] + "/" + sentences[k]] = calc_cosine_similarity(embeddings[j], embeddings[k])
#         with open(cosine_similarity_filename, "wb") as f:
#             pickle.dump(cosine_similarities, f)
#     return cosine_similarities

# cos_sims = generate_cosine_similarities()





# print("finished with cosine similarites")

# print(len(cos_sims))

# def cluster_by_cos_sim(threshold = .5):
#     cluster_by_cos_sim_filename = "clusters_cos_sim_" + str(len(cos_sims)) + "_" + str(threshold) + ".pkl"
#     if os.path.isfile(cluster_by_cos_sim_filename):
#         with open (cluster_by_cos_sim_filename, "rb") as f:
#             sentence_clusters = pickle.load(f)
#             return sentence_clusters
#     ungrouped_sentences = sentences.copy()
#     sentence_clusters = []
#     while len(ungrouped_sentences) > 0:
#         #create new group for this index
#         start_sentence = ungrouped_sentences[0]
#         new_cluster = [start_sentence]
#         new_ungrouped_sentences = []
#         #loop through each ungrouped sentence and compare to ungrouped_sentecs[0] to see if they are similar
#         for sentence in ungrouped_sentences[1:]:
#             key = start_sentence + "/" + sentence
#             if key not in cos_sims:
#                 key = sentence + "/" + start_sentence
#             if key not in cos_sims:
#                 continue
#             # if similar: then add to new group and remove from ungrouped
#             if cos_sims[key] < threshold:
#                 new_cluster.append(sentence)
#             else:
#                 new_ungrouped_sentences.append(sentence)
#         ungrouped_sentences = new_ungrouped_sentences
#         #save new group
#         sentence_clusters.append(new_cluster)
#     with open(cluster_by_cos_sim_filename, "wb") as f:
#         pickle.dump(sentence_clusters, f)
#     return sentence_clusters

# clusters = cluster_by_cos_sim(.5)

# print("finished with clustering")





def draw(labels, name):
     # Apply PCA to reduce dimensionality to 3D for better visualization
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    plt.ion()

    # Plot the 2D PCA results with K-Means cluster labels
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    colors = plt.get_cmap('tab20', len(unique_labels))

    for label in unique_labels:
        class_member_mask = (labels == label)
        xy = embeddings_pca[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=5, label=f'Cluster {label+1}', color=colors(label))

    plt.title('PCA of Sentence Embeddings with ' + name + ' Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

    # Keep the plot displayed and allow the script to continue running
    plt.pause(0.001)

def k_means_cluster(embeddings,  num_clusters=7):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    return labels

def GMM_cluster(embeddings, num_clusters=7):
    gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(embeddings)
    labels = gmm.predict(embeddings)
    return labels

def agglomerative_cluster(embeddings, num_clusters=7):
    # Use Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
    labels = agg_clustering.fit_predict(embeddings)
    return labels



# def find_optimal_gmm_clusters(embeddings, max_clusters=10):
#     bics = []
#     aics = []
#     cluster_range = range(1, max_clusters + 1)
#     print ("Cluster range: " + str(cluster_range))
#     counter = 0

#     for n_clusters in cluster_range:
#         print(counter)
#         counter+=1
#         gmm = GaussianMixture(n_components=n_clusters, random_state=0)
#         gmm.fit(embeddings)
#         bics.append(gmm.bic(embeddings))
#         aics.append(gmm.aic(embeddings))

#     optimal_clusters_bic = cluster_range[np.argmin(bics)]
#     optimal_clusters_aic = cluster_range[np.argmin(aics)]

#     # Plot BIC and AIC
#     plt.figure(figsize=(10, 7))
#     plt.plot(cluster_range, bics, label='BIC')
#     plt.plot(cluster_range, aics, label='AIC')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Score')
#     plt.title('BIC and AIC for different numbers of clusters')
#     plt.legend()
#     plt.show()

#     return optimal_clusters_bic, optimal_clusters_aic

# optimal_clusters_bic, optimal_clusters_aic = find_optimal_gmm_clusters(embeddings, max_clusters=10)

# print(f"Optimal number of clusters by BIC: {optimal_clusters_bic}")
# print(f"Optimal number of clusters by AIC: {optimal_clusters_aic}")

def get_top_words_per_cluster(posts, labels, num_clusters=4, top_n=30):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(posts)
    terms = vectorizer.get_feature_names_out()
    
    top_words = {}
    for i in range(num_clusters):
        cluster_posts = X[labels == i]
        mean_tfidf = cluster_posts.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-top_n:]
        top_words[i] = [terms[ind] for ind in top_indices][::-1]
    
    return top_words

# Perform clustering
kmeans_labels = k_means_cluster(embeddings, 4)
gmm_labels = GMM_cluster(embeddings, 4)
agglo_labels = agglomerative_cluster(embeddings, 4)

#Draw plots for each clustering method
draw(kmeans_labels, "K-Means")
print("finished with k means")
draw(gmm_labels, "GMM")
print("finished with GMM")
draw(agglo_labels, "Agglomerative")
print("finished with agglomerative")

# Get top words for K-Means clusters
top_words_kmeans = get_top_words_per_cluster(sentences, kmeans_labels)
print("Top words per cluster for K-Means:")
for cluster, words in top_words_kmeans.items():
    print(f"Cluster {cluster + 1}: {', '.join(words)}")

top_words_gmm = get_top_words_per_cluster(sentences, gmm_labels)
print("Top words per cluster for GMM:")
for cluster, words in top_words_gmm.items():
    print(f"Cluster {cluster + 1}: {', '.join(words)}")

top_words_agglo = get_top_words_per_cluster(sentences, agglo_labels)
print("Top words per cluster for Agglomerative:")
for cluster, words in top_words_agglo.items():
    print(f"Cluster {cluster + 1}: {', '.join(words)}")

input("Press [enter] to end.")



# Cluster similar sentence embeddings for classification (semi-supervised clustering)

# If we have a single tweet to evaluate, check which cluster is most closely belongs to
