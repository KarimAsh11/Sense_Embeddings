from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resources_path", help="The path of the resources needed")

    return parser.parse_args()


def plot_pca(model, keys):
	"""
	plot sense embeddings - PCA.

	params:
	    model   (w2v object)
	    keys 	(list)
	"""
	vecs = model[keys]

	pca = PCA(n_components=2)
	result = pca.fit_transform(vecs)

	pyplot.scatter(result[:, 0], result[:, 1])
	words = keys
	for i, word in enumerate(words):
		pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
	pyplot.show()



def tsne_help(model, keys, topn):
	"""
	prepares data for t-SNE.

	params:
	    model   (w2v object)
	    keys 	(list)
	    topn    (int)
	"""
	embedding_clusters = []
	word_clusters = []
	for word in keys:
		embeddings = []
		words = []
		for similar_word, _ in model.most_similar(word, topn=topn):
			words.append(similar_word)
			embeddings.append(model[similar_word])
		embedding_clusters.append(embeddings)
		word_clusters.append(words)

	embedding_clusters = np.array(embedding_clusters)
	n, m, k = embedding_clusters.shape
	tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
	embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

	return embeddings_en_2d, word_clusters



def plot_tsne(title, model, labels, a, filename=None):
	"""
	plot sense embeddings - t-SNE.

	params:
	    title    (str)
	    labels 	 (list)
	    a        (float)
	    filename (str)
	"""
	embedding_clusters, word_clusters = tsne_help(model, labels, 5)
	plt.figure(figsize=(16, 9))
	colors = cm.rainbow(np.linspace(0, 1, len(labels)))
	for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
		x = embeddings[:, 0]
		y = embeddings[:, 1]
		plt.scatter(x, y, c=color, alpha=a, label=label)
		for i, word in enumerate(words):
			plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
				textcoords='offset points', ha='right', va='bottom', size=8)
	plt.legend(loc=4)
	plt.title(title)
	plt.grid(True)
	if filename:
		plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
	plt.show()


if __name__ == '__main__':
	args = parse_args()
	resources_path = args.resources_path
	emb_path       = resources_path+"/embeddings.vec"
	model      = KeyedVectors.load_word2vec_format(emb_path, binary=False)
	keys       = ["bank_bn:00008364n"]

	plot_tsne("t-SNE", model, keys, 0.7)
