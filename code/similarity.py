from utility import load_pickle, save_pickle, read, write, readlines, write_str
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from argparse import ArgumentParser
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resources_path", help="The path of the resources needed")

    return parser.parse_args()


def read_combined(comb_path, words_path, score_path):
	"""
	Read combined.tsv to retrieve pair of words and scores.

	params:
	    comb_path  (str)
	    words_path (str)
	    score_path (str)
	"""
	words = []
	score = []
	file = readlines(comb_path)
	file.pop(0)

	for line in file:
		l = line.rstrip().split('\t')

		words.append(l[:2])
		score.append(l[-1])

	save_pickle(words, words_path)
	save_pickle(score, score_path)
	print("Finished saving combined")


def correlation(model, words, gold, closest, cosine):
	"""
	Calculate spearman correlation between predictions and human score.

	params:
	    model    (Word2Vec object)
	    words    (list)
	    gold     (list)
	    closest  (boolean)
	    cosine   (boolean)

	returns: float
	"""
	print("calculating correlation")

	if closest:
		if cosine:
			pred  = max_cos_similarity(model, words)
		else:
			pred  = max_tan_similarity(model, words)

	else:
		if cosine:
			pred  = weighted_cos_similarity(model, words)
		else:
			pred  = weighted_tan_similarity(model, words)

	return spearmanr(gold, pred)


def get_senses(word, sense_list):
	"""
	Retrieve senses for the given word.

	params:
	    word        (str)
	    sense_list  (list)

	returns: list
	"""
	word_senses = []
	for sense in sense_list:
		if '_bn:' in sense and sense.split("_bn:")[0] == word.lower():
			word_senses.append(sense)
	return word_senses


def cos_helper(s1, s2):
	"""
	Calculate cosine distance between each pair of words using numpy.

	params:
	    s1   (array)
	    s2   (array)

	returms: float
	"""
	dot = np.dot(s1, s2)
	norm1 = np.linalg.norm(s1)
	norm2 = np.linalg.norm(s2)
	cos = dot / (norm1 * norm2)

	return cos 


def tanimoto(s1, s2):
	"""
	Calculate Tanimoto distance between each pair of words using numpy.

	params:
	    s1   (array)
	    s2   (array)

	returms: float
	"""
	dot = np.dot(s1, s2)
	norm1 = np.linalg.norm(s1)
	norm2 = np.linalg.norm(s2)
	cos = dot / ((norm1**2) + (norm2**2) - dot)

	return cos


def max_cos_similarity(model, words):
	"""
	Calculate closest similarity between each pair of words - COS.

	params:
	    model    (Word2Vec object)
	    words   (list)

	returns: list
	"""
	sense_list = list(model.wv.vocab)
	cos_list = []

	for pair in words:

		senses_1 = get_senses(pair[0], sense_list)
		senses_2 = get_senses(pair[1], sense_list)

		score = -1.

		for s_1 in senses_1:
			for s_2 in senses_2:
				score  = max(score, cos_helper(model[s_1],model[s_2]))
		cos_list.append(score)

	return cos_list


def max_tan_similarity(model, words):
	"""
	Calculate closest similarity between each pair of words - Tanimoto.

	params:
	    model    (Word2Vec object)
	    words   (list)

	returns: list
	"""
	sense_list = list(model.wv.vocab)
	cos_list = []

	for pair in words:

		senses_1 = get_senses(pair[0], sense_list)
		senses_2 = get_senses(pair[1], sense_list)

		score = -1.

		for s_1 in senses_1:
			for s_2 in senses_2:
				score  = max(score, tanimoto(model[s_1],model[s_2]))
		cos_list.append(score)

	return cos_list


def sk_cos_similarity(model, words):
	"""
	Calculate max cosine similarity between each pair of words using sklearn cos similarity.

	params:
	    model    (Word2Vec object)
	    words   (list)

	returns: list
	"""
	sense_list = list(model.wv.vocab)
	cos_list = []

	for pair in words:

		senses_1 = get_senses(pair[0], sense_list)
		senses_2 = get_senses(pair[1], sense_list)

		score = -1.

		for s_1 in senses_1:
			for s_2 in senses_2:
				score  = max(score, cosine_similarity(model[s_1].reshape(1, -1),model[s_2].reshape(1, -1)))
		cos_list.append(score)

	return cos_list



def gensim_cos_similarity(model, words):
	"""
	Calculate max cosine similarity between each pair of words using gensim cos similarity.

	params:
	    model    (Word2Vec object)
	    words   (list)

	returns: list
	"""
	sense_list = list(model.wv.vocab)
	cos_list = []

	for pair in words:

		senses_1 = get_senses(pair[0], sense_list)
		senses_2 = get_senses(pair[1], sense_list)

		score = -1.

		for s_1 in senses_1:
			for s_2 in senses_2:
				score  = max(score, model.wv.similarity(s_1, s_2))
		cos_list.append(score)

	return cos_list



def weighted_cos_similarity(model, words):
	"""
	Calculate weighted cosine similarity between each pair of words using gensim cos similarity.

	params:
	    model    (Word2Vec object)
	    words   (list)

	returns: list
	"""
	sense_list = list(model.wv.vocab)
	cos_list = []

	for pair in words:

		senses_1 = get_senses(pair[0], sense_list)
		senses_2 = get_senses(pair[1], sense_list)

		Total_Freq_S1 = get_total_freq(model, senses_1)
		Total_Freq_S2 = get_total_freq(model, senses_2)

		if len(senses_1)==0 or len(senses_2)==0:
			cos_list.append(0)
			continue

		score = 0.

		for s_1 in senses_1:
			for s_2 in senses_2:
				dom1      = dom_sense(model, s_1, Total_Freq_S1)
				dom2      = dom_sense(model, s_2, Total_Freq_S2)
				distance  = model.wv.similarity(s_1, s_2)
				score    += dom1*dom2*distance 
		cos_list.append(score)

	return cos_list


def weighted_tan_similarity(model, words):
	"""
	Calculate weighted similarity between each pair of words bases on Tanimoto distance.

	params:
	    model    (Word2Vec object)
	    words   (list)

	returns: list
	"""
	sense_list = list(model.wv.vocab)
	cos_list = []

	for pair in words:

		senses_1 = get_senses(pair[0], sense_list)
		senses_2 = get_senses(pair[1], sense_list)

		Total_Freq_S1 = get_total_freq(model, senses_1)
		Total_Freq_S2 = get_total_freq(model, senses_2)

		if len(senses_1)==0 or len(senses_2)==0:
			cos_list.append(0)
			continue

		score = 0.

		for s_1 in senses_1:
			for s_2 in senses_2:
				dom1      = dom_sense(model, s_1, Total_Freq_S1)
				dom2      = dom_sense(model, s_2, Total_Freq_S2)
				distance  = tanimoto(model[s_1], model[s_2])
				score    += dom1*dom2*distance 
		cos_list.append(score)

	return cos_list


def dom_sense(model, sense, total_freq):
	"""
	Calculate dominance of sense.

	params:
	    model   	 (w2v object)
	    sense   	 (str)
	    total_freq   (int)

	returms: float
	"""
	return (model.wv.vocab[sense].count)/total_freq


def get_total_freq(model, senses):
	"""
	Calculate total frequency of senses.

	params:
	    model    (Word2Vec object)
	    senses   (list)

	returms: float
	"""
	return sum([model.wv.vocab[sense].count for sense in senses])



if __name__ == '__main__':
	args = parse_args()
	resources_path = args.resources_path
	comb_path 	   = resources_path+"/combined.tab"
	words_path     = resources_path+"/words.pickle"
	score_path     = resources_path+"/scores.pickle"
	emb_path       = resources_path+"/embeddings.vec"

	read_combined(comb_path, words_path, score_path)

	words = load_pickle(words_path) 
	gold  = load_pickle(score_path) 

	model  = KeyedVectors.load_word2vec_format(emb_path, binary=False)

	cor = correlation(model, words, gold, True, True)
	print("max cos", cor)

