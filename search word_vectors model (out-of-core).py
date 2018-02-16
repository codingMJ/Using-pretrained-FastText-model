'''
Using trained FastText model for out-of-core search.
'''

from gensim.models.wrappers import FastText
from gensim.matutils import cossim
import numpy as np
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter

# calculate cosine similarity between vectors
# and vectors can be either arrays or lists.
def cos_sim(a, b):

	return dot(a, b)/(norm(a)*norm(b))

# Returns True if token is in the trained model vocabulary
# and returns False if not found
def is_token_in_vocab(search_token, wordvec_model_path):
	with open(wordvec_model_path, 'rb') as infile:
		for line in infile:
			line_decoded = line.decode("utf-8")
			word, vec_s  = line_decoded.strip().split(' ', 1)
			if search_token == word:
				return True

	return False

# Returns array vector of token from trained model
# and returns None if token is not in the trained model vocabulary
def get_token_vector(search_token, wordvec_model_path):
	search_token_vec = None
	with open(wordvec_model_path, 'rb') as infile:
		for line in infile:
			line_decoded = line.decode("utf-8")
			word, vec_s = line_decoded.strip().split(' ', 1)
			if search_token == word:
				search_token_vec = np.array([float(v) for v in vec_s.split(' ')])
				break
				
	return search_token_vec

# Gets top closest tokens to search token (e.g. token_vec) and default top = 2
# returns tuple of 3 lists:
# 1 - list of closest tokens, 
# 2 - list of their corresponding vectors
# 3 - list of their corresponding cosine distances to the sarch token
def get_top_similar(token_vec, wordvec_model_path, top_n=2):
	c = 0
	top_n_word2vec = {}
	token_list = []
	vec_list = []
	sim_list = []

	with open(wordvec_model_path, 'rb') as infile:
		for line in infile:
			line_decoded = line.decode("utf-8")
			word, vec_s = line_decoded.strip().split(' ', 1)
			vec = np.array([float(v) for v in vec_s.split(' ')])

			try:
				sim = cos_sim(token_vec, vec)
			except Exception as e:
				# print(word, ' - ', token_vec)
				# print(word, ' - ', vec)
				# print(e)
				# print(type(token_vec))
				# print(type(vec))
				continue

			sim_list.append(sim)
			token_list.append(word)
			vec_list.append(vec)

			c = c + 1
			if c > top_n:
				idx, sim_value = min(enumerate(sim_list), key=itemgetter(1))
				sim_list   = sim_list[0:idx] + sim_list[(idx+1):]
				token_list = token_list[0:idx] + token_list[(idx+1):]
				vec_list   = vec_list[0:idx] + vec_list[(idx+1):]

			

	return 	token_list, vec_list, sim_list




if __name__ == '__main__':
	wordvec_model_path = 'fastext model - wiki_en/wiki.en.vec'

	# search tokens
	ls = ['adidas', 'hammer', 'gb']
	ls = ['brazzers','wellsfargo']


	# get top similar tokens from trained word vector model
	for t in ls:
		search_token = t
		print(search_token)

		if is_token_in_vocab(search_token, wordvec_model_path):
			print('Start ...')
			v = get_token_vector(search_token, wordvec_model_path)

			token_list, vec_list, sim_list = get_top_similar(v, wordvec_model_path, top_n=200)
			print('Closest tokens: ', token_list)
			print('Cosine distances: ', sim_list)
			print('=' * 40)












