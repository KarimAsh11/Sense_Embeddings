import os
from utility import load_pickle

class MySentences(object):
	"""
	Streams data to model to avoid loading all at once.
	
	params:
		dirname (string)
	"""
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for root, subdir, fileList in os.walk(self.dirname):
			for fname in filter(lambda fname: fname.endswith('.pickle'), fileList):
				for line in load_pickle(os.path.join(root, fname)):
					yield line	