from gensim.models.wrappers import FastText
from gensim.matutils import cossim
import numpy as np
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter



if __name__ == '__main__':
	wordvec_model_path = 'wiki.en.vec'
	infile = open(wordvec_model_path, 'rb')
	outfile_name = ''wiki_en.txt''


	#get most similar vectors to search token
	c = 0
	with open(outfile_name, 'w') as outfile:
		for line in infile:
			line_decoded = line.decode("utf-8")
			word, vec_s = line_decoded.strip().split(' ', 1)

			if c < 4:
				vec = np.array([float(v) for v in vec_s.split(' ')])

				try:
					sim = cos_sim(search_token_vec, vec)
					print(sim)
				except Exception as e:
					print(len(vec), ' - ', len(search_token_vec))

			    # prepare text line for writing into file
				line_values = [word, ':']
				for x in vec:
					line_values.extend(str(x))
				line_values.extend('\n')

				# write text line to file
				line_values = ''.join(line_values)
				outfile.write(line_values)

			c += 1

		infile.close()












