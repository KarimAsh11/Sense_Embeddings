import nltk
import string
from utility import read, write, load_pickle, save_pickle
from parse_utility import anchor, get_map, get_map_dict, iter_help, lemma_syns
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm
from lxml import etree
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def eurosense_parse(input_path, out_path, resources_path):
	"""
	Parses EuroSense high precision.

	params:
	    input_path		(str)
	    out_path		(str)
	    resources_path	(str)
	"""
	i=0
	corp            = []
	syn             = []
	syns_dict       = defaultdict(list)
	map_path		= resources_path+"/bn2wn_mapping.txt"
	file_path		= resources_path+"/"+input_path
	syns_map        = get_map_dict(map_path)
	
	print("Parsing XML file ...")

	for event, element in etree.iterparse(file_path, tag="sentence"):
		clean_sentence      = []
		base_anchor_list    = []
		syns_dict.clear()
		i+=1

		if event == "end":
			sentence, base_anchor_list, syns_dict = iter_help(element, syns_dict, syns_map)

		if len(sentence):
			sentence       = anchor(sentence, base_anchor_list)
			clean_sentence = lemma_syns(sentence, syns_dict)	
			corp.append(clean_sentence)

		element.clear()

		if i%10000==0:
			print("EXAMPLE: ", i)
		
	print("Writing output file ...")
	save_pickle(corp, out_path)


if __name__ == '__main__':
    args = parse_args()
    eurosense_parse(args.input_path, args.output_path, args.resources_path)

