from utility import read, write, load_pickle
from stream import MySentences
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from argparse import ArgumentParser
import logging

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")
    parser.add_argument("euro_path", help="The name of the EuroSense corpus in use")

    return parser.parse_args()


def train_w2v_stream(input_path, output_path, resources_path, euro_path):
	"""
	Train Word2V model on stream of data.

	params:
	    input_path		(str)
	    output_path		(str)
	    resources_path	(str)
	"""
	corpus_path = resources_path+"/"+input_path
	model_path  = resources_path+"/"+output_path
	euro_path  = resources_path+"/"+euro_path

	corpus    = load_pickle(euro_path)

	sentences = MySentences(corpus_path)

	logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

	cores = cpu_count()

	w2v_model = Word2Vec(min_count=1, window=10, size=500, sample=1e-3, workers=cores-1, negative=10, sg=0)
	w2v_model.build_vocab(corpus, progress_per=10000)

	w2v_model.build_vocab(sentences, progress_per=10000, update=True)
	w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1, compute_loss=True)
	logging.info('Training phase over ...')


	w2v_model.wv.save_word2vec_format(model_path, binary=False)
	logging.info('Saving weights phase over ...')



if __name__ == '__main__':
    args = parse_args()
    train_w2v_stream(args.input_path, args.output_path, args.resources_path, args.euro_path)
