import pickle

def read(file_path, encoding='utf8'):
	"""
	Read input file.

	params:
	    file_path (str)

	returns: str
	"""
	with open(file_path, 'r') as f:
		return f.read()

def readlines(file_path, encoding='utf8'):
	"""
	Read input file.

	params:
	    file_path (str)

	returns: str
	"""
	with open(file_path, 'r') as f:
		return f.readlines()


def write(content, file_path, encoding='utf8'):
    """
    Write to output file

    params:
        content List[List[str]]
        file_path (str)
        encoding (str) [default='utf8']
    """
    with open(file_path, 'w', encoding=encoding) as f:
        for line in content:
            f.write(''.join(line))
            if line != content[-1]:
                f.write('\n')


def write_str(content, file_path, encoding='utf8'):
    """
    Write to output file

    params:
        content List[List[str]]
        file_path (str)
        encoding (str) [default='utf8']
    """
    with open(file_path, 'w', encoding=encoding) as f:
        for line in content:
            f.write(''.join(str(line)))
            if line != content[-1]:
                f.write('\n')



def save_pickle(dictionary, pickle_file):
    """
    Pickle dictionary.

    params:
        dictionary (dict)
        pickle_file (str)
    """
    with open(pickle_file, 'wb') as h:
        pickle.dump(dictionary, h, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(pickle_file):
	"""
	Pickle dictionary.

	params:
	    pickle_file (pickle)
	"""
	with open(pickle_file, 'rb') as h:
		return pickle.load(h)


def clean_vec(vec_path, clean_vec_path):
	"""
	Keeps sense embeddings only.

	params:
	    vec_path       (str)
	    clean_vec_path (str)
	"""
	vectors    = read(vec_path).splitlines()
	sense_vec  = []

	_, emb_size   = vectors[0].split()

	for vec in vectors:
		if '_bn:' in vec:
			sense_vec.append(vec)

	first_line = str(len(sense_vec)) + " " + emb_size
	sense_vec.insert(0, first_line)


	write(sense_vec, clean_vec_path)

