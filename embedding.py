from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample sentences
sentences = [
    'My name is Sandeep',
    'Sandeep lives in new Delhi',
    'Sandeep loves technology',
]

# Tokenize the sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Create Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# Get the word vector for a specific word
word_vector = model.wv['sandeep']

print("Embedding for 'Sandeep':", word_vector)
