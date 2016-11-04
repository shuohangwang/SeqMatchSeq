
#Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

#Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

#This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.


task=$1
GloVe="data/glove/glove.840B.300d.txt"
WikiQA="data/wikiqa/WikiQACorpus.zip"
SQuAD="data/squad/train-v1.1.json"
SNLI="data/snli/snli_1.0.zip"


if [ ! -f "$GloVe" ]; then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/glove
	unzip -o -d data/glove/ data/glove/glove.840B.300d.zip
fi;


if [ "$task" = "wikiqa" ]; then
	if [ ! -d "data" ]; then 
		mkdir data 
	fi;
	if [ ! -d "data/wikiqa" ]; then
		mkdir data/wikiqa/ 
        fi;
	if [ ! -d "data/wikiqa/sequence" ]; then
		mkdir data/wikiqa/sequence
	fi;

	if [ -f "$WikiQA" ]; then
		unzip -o -d data/wikiqa/ data/wikiqa/WikiQACorpus.zip
	else
		echo "!!!!!!!!Please dowload the file \"WikiQACorpus.zip\" to the path data/wikiqa/ through address: https://www.microsoft.com/en-us/download/details.aspx?id=52419"
	fi;
elif [ "$task" = "squad" ]; then
	if [ ! -f "$SQuAD" ]; then
		wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -P data/squad
		wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P data/squad
	fi;
	if [ ! -d "data/squad/sequence" ]; then 
		mkdir data/squad/sequence
	fi;
	curl https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ > trainedmodel/evaluation/squad/evaluate-v1.1.py
	python preprocess.py squad
elif [ "$task" = "snli" ]; then
    if [ ! -f "$SNLI" ]; then
		wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip -P data/snli
		unzip -o -d data/snli/ data/snli/snli_1.0.zip
	fi;
	if [ ! -d "data/snli/sequence" ]; then 
		mkdir data/snli/sequence
	fi;
	python preprocess.py snli
fi;
