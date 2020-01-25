
import io
import pickle
import settings
import json
import clear
from text_cleaning import clean_text
import numpy as np

try:
	clear.clear_cache(clear_model = settings.clear_model)
except:
	pass

#lines_path = 'data/movie_lines.txt' 
#conv_path = 'data/movie_conversations.txt'

with io.open(settings.lines_path, 'rb') as f:
  lines = f.read().decode(encoding="ascii", errors="ignore").split("\n")

with io.open(settings.conv_path, 'rb') as f:
  conv_lines = f.read().decode(encoding="ascii", errors="ignore").split("\n")

print()

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
    
data = []

for conv in convs[:int(len(convs))]:
    for i in range(len(conv)):
        #print(conv[i])
        data.append(id2line[conv[i]].lower())

min_line_length = settings.min_line_length
max_length  = settings.max_length
dictionary_length  = settings.dictionary_length

print("Cleaning conversations & Creating word dictionary.....")

all_sentences = []
for sentence in data:
    for sub_sentence in sentence.split("."):
        cleaned = clean_text(sub_sentence).split()

        if len(cleaned) >= min_line_length and len(cleaned)<max_length:
            all_sentences.append(" ".join(cleaned))

sentences_samples = all_sentences[:settings.sentences_samples] #training of few number of samples

vocab = {}
for sentence in sentences_samples:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

vocab_sort  = {letter:ind+1 for ind,letter in enumerate(sorted(vocab, key=vocab.get, reverse=True))}

print("Length of unique vocabulary: {}".format(len(vocab_sort)))

min_count = settings.min_count
print("vocabulary with count >= {}: {}".format(min_count,sum(np.array(list(vocab.values()))>=min_count)))

word_index = {e:i for e,i in vocab_sort.items() if i <= dictionary_length-2}
word_index["<PAD>"] = 0
word_index["<OOV>"] = dictionary_length-1


corpus = []
labels = []
for sentence in sentences_samples:
    sen = []
    for word in sentence.split():
        inde = word_index.get(word)
        if inde != None:
            sen.append(inde)
        else:
            sen.append(word_index["<OOV>"])
    for i in range(len(sen)-1):
        corpus.append(sen[:i+1])
        labels.append(sen[i+1])

with open("vocab.pkl", "wb") as f:
 	pickle.dump(vocab_sort, f)

with open("word_index_dic.pkl", "wb") as f:
 	pickle.dump(word_index, f)

with open('sentences.pkl', 'wb') as f:
	pickle.dump(sentences_samples, f)

with open('indexed_sentences.pkl', 'wb') as f:
    pickle.dump(corpus, f)

with open('indexed_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print("Length of corpus: {}".format(len(corpus)))
