# Scripts
euro_parse.py parses EuroSense - high precision and preprocesses the data

sew_parse.py parses SEW - conservative version and preprocesses the data

gen_model.py trains a Word2Vec model - Data does not saturate RAM

train_sense.py trains a Word2Vec model - Streams data sample by sample - For large datasets 

parse_utility.py contains the helper functions for parsing and preprocessing

utility.py contains the utility functions

stream.py contains iterator responsible for streaming the data in case of too large dataset

similarity.py provides different strategies for calculating the similarity (closest and weighted), diifferent distances (cosine and Tanimoto), and spearman correlation function

plot.py contains functions responsible for generating the t-SNE and PCA plots
