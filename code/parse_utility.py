import re, string
from nltk.corpus import stopwords


def get_map(file_path):
	"""
	Get syns map.

	params:
	    file_path (str)

	returns: list
	"""
	with open (file_path, 'r') as f:
	    syns = [row.split('\t')[0] for row in f]
	    return syns

def get_map_dict(file_path):
	"""
	Get syns map.

	params:
	    file_path (str)

	returns: dict
	"""
	with open (file_path, 'r') as f:
	    syns = {row.split('\t')[0]:1 for row in f}
	    return syns

def anchor(sen, anchor):
	"""
	replace anchor with spaceless anchor.

	params:
	    sen (str)
	    anchor (list)

	returns: str
	"""
	anchor.sort(key=len, reverse=True)

	for i in anchor:
		new=i.replace(" ", "_")
		sen=sen.replace(i, new, 1)

	return sen

def preprocess(sen):
	"""
	Preprocess sentence.

	params:
	    sen (str)

	returns: str
	"""

	sen = sen.translate(str.maketrans('', '', string.punctuation))   
	sen = re.sub(' +', ' ',sen)

	return sen


def iter_help(element, syns_dict, syns_map):
	"""
	prepares sentence and sense dictionary - eurosense dataset.

	params:
	    element (obj)
	    syns_dict(dict)
	    syns_map(dict)

	returns: str, list, dict
	"""
	base_anchor_list = []
	sentence = ""
	for el in element.iter():
		if element.text is not None:
			if el.tag=="text" and el.attrib['lang']=="en":
				if el.text == None:
					return sentence, base_anchor_list, syns_dict
					continue
				else:
					sentence = preprocess(el.text)

			if el.tag=="annotation" and el.attrib['lang']=="en":
				if syns_map.get(el.text) is not None:
					base_anchor = el.attrib['anchor'].replace("-", "")
					new_anchor  = base_anchor.replace(" ", "_")
					base_anchor_list.append(base_anchor)

					lemma = el.attrib['lemma'].replace(" ", "_").replace("-", "_").lower()
					syns_dict[new_anchor].append(lemma+"_"+el.text)
		else:
			return sentence, base_anchor_list, syns_dict

	return sentence, base_anchor_list, syns_dict



def lemma_syns(sen, syn_dict):
	"""
	replace anchor with lemma_syns.

	params:
	    sen (string)
	    syn_dict (dict)

	returns: list
	"""
	clean_sentence=[]
	sentence = sen.split()

	blacklist = stopwords.words('english')

	for word in sentence:

		if word in syn_dict:
			if len(syn_dict[word]) != 0:
				clean_sentence.append(syn_dict[word].pop(0)) 
			else:
				clean_sentence.append(word.lower())

		elif word.isdigit():
			clean_sentence.append("!")

		elif word.lower() not in blacklist and len(word) > 2:
			clean_sentence.append(word.lower())

		# elif not word.isalpha():
		# 	print("CATCH WRONG WORD", word)

	return clean_sentence



def sew_help(element, syns_dict, syns_map):
	"""
	prepares sentence and sense dictionary - SEW dataset.

	params:
	    element (obj)
	    syns_dict(dict)
	    syns_map(dict)

	returns: str, list, dict
	"""
	sentence = ""
	base_mention_list = []

	for el in element.iter():
		if element.text is not None:
			if el.tag=="text":
				if el.text == None:
					print("TXT EMPTY")
					return sentence, base_anchor_list, syns_dict
					continue

				else:
					print("Preprocess TXT")
					sentence = preprocess(el.text)

			if el.tag=="annotations":
				for child in el:
					if child.text:
						bn_syns = child.xpath('babelNetID')[0].text
						if syns_map.get(bn_syns) is not None:
							mention = child.xpath('mention')[0].text
							if mention is not None:
								base_mention = mention.replace("-", "")
								new_mention  = base_mention.replace(" ", "_")
								base_mention_list.append(base_mention)

								lemma = mention.replace(" ", "_").replace("-", "_").lower()
								syns_dict[new_mention].append(lemma+"_"+bn_syns)
							else:
								continue

		else:
			return sentence, base_mention_list, syns_dict

	return sentence, base_mention_list, syns_dict



def join_dataset(root, path):
	"""
	Creates joint list of datasets in directory.

	params:
	    root	(str)
	    path	(str)
	"""
	dataset = []
	for root, subdir, fileList in os.walk(root):
		for fname in filter(lambda fname: fname.endswith('.pickle'), fileList):
			dataset.extend(load_pickle(os.path.join(root, fname)))

	save_pickle(dataset, path)



