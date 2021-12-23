import nltk
import string
from utility import read, write, load_pickle, save_pickle
from parse_utility import anchor, get_map, get_map_dict, iter_help, lemma_syns, sew_help
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm
from lxml import etree
from glob import iglob
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def sew_parse(input_path, out_path, resources_path):
	"""
	Parses SEW conservative version.

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

	for i, xml_file in tqdm(enumerate(iglob(dir_path)), desc="PROCESSING SEW..."):
		wiki_page               = []
		for event, element in etree.iterparse(xml_file, tag="wikiArticle", encoding='utf-8', recover=True):
			clean_sentence      = []
			base_anchor_list    = []
			syns_dict.clear()
			i+=1

			if event == "end":
				sentence, base_anchor_list, syns_dict = sew_help(element, syns_dict, syns_map)

			if len(sentence):
				sentence       = anchor(sentence, base_anchor_list)
				clean_sentence = lemma_syns(sentence, syns_dict)	

			wiki_page.extend(clean_sentence)
			element.clear()
		corp.append(wiki_page)

		if i%50000==0:
			print("File: ", i)

		if i%200000==0:
			print("Saving", i)
			pickle_ckpt = out_path + "_ckpt_"+str(i)+pic_path
			save_pickle(corp, pickle_ckpt)
			corp.clear()
		
	pickle_final= out_path + "_final"+pic_path
	save_pickle(corp, pickle_final)


if __name__ == '__main__':
    args = parse_args()
    sew_parse(args.input_path, args.output_path, args.resources_path)
