'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import string
from nltk.tokenize import word_tokenize

fp = open(sys.argv[1],'r')
line = fp.readline()
js = json.loads(line)
fpr = open(sys.argv[2],'r')

predictions = {}
for c in js["data"]:
	for p in c["paragraphs"]:
		context = p["context"]
		context_word = word_tokenize(context)
		for qa in p["qas"]:
			question_id = qa["id"]

			pred = fpr.readline().strip()
			newpred = []
			pre_word = ''
			words = pred.split()
			for word in words:
				word = word.replace("``", "\"")
				word = word.replace("''", "\"")
				if pre_word != '' :
					if newpred[-1]+word in context:
						newpred[-1] = newpred[-1]+word
					else:
						newpred.append(word)
				else:
					newpred.append(word)
				pre_word = word
			predictions[question_id] = ' '.join(newpred)

predictions_js = json.dumps(predictions)
fpw = open(sys.argv[3], 'w')
fpw.write(predictions_js)
fpw.close()
fp.close()
fpr.close()
