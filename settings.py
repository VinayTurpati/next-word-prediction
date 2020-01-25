
import os

BASE_PATH = os.path.dirname(__name__)
lines_path = os.path.join(BASE_PATH, "data/movie_lines.txt")
conv_path = os.path.join(BASE_PATH, "data/movie_conversations.txt")

clear_model = True

#Data preparation parameters
min_line_length = 3
max_length  = 20
dictionary_length = 5000	#length of dictionary
sentences_samples = 24000	#total no. of samples 400000+
min_count = 3	#vocabulary minimum count 

epoch = 5
step = 2048
batch_size = 128

padding_type = "post"
truncating_type = "post"

test_samples = ["you", "do you have any", "i'm preparing for", "people are", "how amazing is", "do you pretending that you are not"]