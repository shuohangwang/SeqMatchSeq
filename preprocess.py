'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import sys
def prepSNLI():
	filenames = ['dev', 'test', 'train']
	labelDict = {'neutral':1, 'entailment':2, 'contradiction':3, '-':0}

	for filename in filenames:
		print ('preprossing ' + filename + '...')
		fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
		count = 0
		fpr.readline()
		fpw = open('data/snli/sequence/'+filename+'.txt', 'w')
		for line in fpr:
			sentences = line.strip().split('\t')
			if sentences[0] == '-':
				continue

			tokens = sentences[1].split(' ')
			tokens = [token for token in tokens if token != '(' and token != ')']
			fpw.write(' '.join(tokens)+'\t')
			tokens = sentences[2].split(' ')
			tokens = [token for token in tokens if token != '(' and token != ')' ]
			fpw.write(' '.join(tokens)+'\t')
			fpw.write(str(labelDict[sentences[0]])+'\n')
			count += 1
		fpw.close()
		fpr.close()
	print ('SNLI preprossing finished!')
def prepSQuAD():
	reload(sys)
	sys.setdefaultencoding('utf-8')
	import json
	from nltk.tokenize import word_tokenize
	count = 0
	filenames = ['dev', 'train']
	for filename in filenames:
		fpr = open("data/squad/"+filename+"-v1.1.json", 'r')
		line = fpr.readline()
		js = json.loads(line)
		fpw = open("data/squad/sequence/"+filename+".txt", 'w')
		for c in js["data"]:
			for p in c["paragraphs"]:
				context = p["context"].split(' ')
				context_char = list(p["context"])
				context_pos = {}
				for qa in p["qas"]:

					question = word_tokenize(qa["question"])

					if filename == 'train':
						for a in qa['answers']:
							answer = a['text'].strip()
							answer_start = int(a['answer_start'])

						#add '.' here, just because NLTK is not good enough in some cases
						answer_words = word_tokenize(answer+'.')
						if answer_words[-1] == '.':
							answer_words = answer_words[:-1]
						else:
							answer_words = word_tokenize(answer)

						prev_context_words = word_tokenize( p["context"][0:answer_start ] )
						left_context_words = word_tokenize( p["context"][answer_start:] )
						answer_reproduce = []
						for i in range(len(answer_words)):
							if i < len(left_context_words):
								w = left_context_words[i]
								answer_reproduce.append(w)
						join_a = ' '.join(answer_words)
						join_ar = ' '.join(answer_reproduce)

						#if not ((join_ar in join_a) or (join_a in join_ar)):
						if join_a != join_ar:
							#print join_ar
							#print join_a
							#print 'answer:'+answer
							count += 1

						fpw.write(' '.join(prev_context_words+left_context_words)+'\t')
						fpw.write(' '.join(question)+'\t')
						#fpw.write(join_a+'\t')

						pos_list = []
						for i in range(len(answer_words)):
							if i < len(left_context_words):
								pos_list.append(str(len(prev_context_words)+i+1))
						if len(pos_list) == 0:
							print join_ar
							print join_a
							print 'answer:'+answer
						assert(len(pos_list) > 0)
						fpw.write(' '.join(pos_list)+'\n')
					else:
						fpw.write(' '.join(word_tokenize( p["context"]) )+'\t')
						fpw.write(' '.join(question)+'\n')

		fpw.close()
	print ('SQuAD preprossing finished!')
if __name__ == "__main__":
	task = sys.argv[1]
	if task == "snli":
		prepSNLI()
	elif task == "squad":
		prepSQuAD()
