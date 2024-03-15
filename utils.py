# Standard library imports
import os
import json
import string
import random
import re
from collections import Counter

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.util import trigrams
from nltk import bigrams
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
import nltk
from nltk.corpus import wordnet
from nltk.corpus import words
# from langdetect import detect, LangDetectException
from gensim.models import Phrases, Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

# Load the English NLP model
nlp_disabled = spacy.load("en_core_web_sm", disable=["parser", "ner", "entity_linker", "attribute_ruler", "lemmatizer"])
nlp = spacy.load("en_core_web_sm")

# Add the sentencizer to the pipeline using the new syntax
nlp.add_pipe('sentencizer')

# Load wordnet
nltk.download('wordnet')

# Load words dataset
nltk.download('words')

# Create a set for faster lookup
english_words = set(words.words())

# Set to half of spaCy's maximum text length
MAX_TEXT_LENGTH = 500000 

#####
# 1. Utility Function for Descriptive Analysis
#####


def load_corpora(base_dir, subdirs, filter=True):
    """
    Load corpora from specified directories, filtering out documents with less than 100 words.

    Parameters:
    - base_dir: The base directory where the corpora are stored.
    - subdirs: A list of subdirectories inside the base directory.

    Returns:
    A dictionary with subdirectory names as keys and a nested dictionary as values.
    The nested dictionary has two keys:
    - 'corpus': a list of documents from the directory.
    - 'filtered_count': the number of documents filtered out due to having less than 100 words.
    """

    # Initialize the corpora and filtered_counts dictionaries
    corpora = {}
    filtered_counts = {}

    # For each subdirectory, load the corpus and count the number of filtered documents
    for subdir in subdirs:
        corpus = []
        filtered_count = 0

        for filename in os.listdir(os.path.join(base_dir, subdir)):
            if filename.endswith('.json'):
                with open(os.path.join(base_dir, subdir, filename), 'r') as file:
                    data = json.load(file)
                    text = data['cleaned_text']
                    if text:
                        word_count = len(text)
                        if filter and word_count >= 100:
                            corpus.append(text)
                        elif not filter:
                            corpus.append(text)
                        else:
                            filtered_count += 1

        # Add the corpus and filtered count to the dictionaries
        corpora[subdir] = corpus
        filtered_counts[subdir] = filtered_count

    return corpora, filtered_counts



def compute_statistics(corpus):
    """
    Compute descriptive statistics for a corpus.

    Parameters:
    - corpus: A list of documents.

    Returns:
    A tuple of descriptive statistics: (min_length, max_length, mean_length, median_length).
    """
    lengths = [len(document.split()) for document in corpus]
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = sum(lengths) / len(lengths)
    median_length = np.median(lengths)
    return min_length, max_length, mean_length, median_length


def total_words_in_corpus(corpus):
    """
    Count words in each document and sum them up.
    """
    return sum([len(document.split()) for document in corpus])


def tokenize_with_spacy(text, chunk_size=500000):
    """Tokenizes a text using spaCy and returns the tokens."""
    
    # If text is small enough, process as a whole
    if len(text) <= chunk_size:
        doc = nlp_disabled(text)
        return [token.text for token in doc]

    # If text is too long, process in chunks
    tokens = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        doc = nlp(chunk)
        tokens.extend([token.text for token in doc])

    return tokens

def process_word(word):
    """Processes individual word and returns cleaned word."""

    # Combine reference/annotation removal and special pattern handling into one regex
    word = re.sub(r'(\.\d+(,\d+)*)$|(\-\d+$)|(\],\d+\-\d+$)', '', word)
    
    # Early return for patterns that make further processing unnecessary
    if re.search(r'\.\d+(,\d+)+$', word):
        return ''
    
    # Remove special characters, efficiently combining character removals
    word = re.sub(r'[^a-zA-Z0-9-]', '', word).strip('.,!;:()[]{}"\'')
    
    return word


def preprocess_sentence_for_analysis(text):
    """
    Process a text for analysis: tokenize, remove stopwords and punctuation.

    Parameters:
    - text: A string.

    Returns:
    A list of tokens.
    """
    # Process the text with SpaCy
    doc = nlp_disabled(text.lower())

    # Filter out stopwords and punctuation, and process words
    tokens = [process_word(token.text) for token in doc if not token.is_stop and not token.is_punct]

    # Filter out short tokens
    tokens = [token for token in tokens if len(token) >= 2]

    return tokens

def preprocess_document_for_analysis(doc):
    """
    Tokenize a document into sentences, then tokenize each sentence into words.
    
    Parameters:
    - doc: A string representing the document.

    Returns:
    A list of lists, where each inner list is a list of tokenized words of a sentence.
    """
    # Process the document with SpaCy to split into sentences
    processed_doc = nlp(doc)
    sentences = [sentence.text for sentence in processed_doc.sents]

    # Now tokenize each sentence
    tokenized_sentences = []
    for sentence in sentences:
        # Tokenize each sentence here (using your existing word tokenization logic)
        sentence_tokens = preprocess_sentence_for_analysis(sentence)  # Replace with your actual word tokenization function
        tokenized_sentences.append(sentence_tokens)

    return tokenized_sentences


# def process_word(word):
#     """Processes individual word and returns cleaned word."""
#     # Remove references or annotations
#     word = re.sub(r'\.\d+(\,\d+)*$', '', word)
#     word = re.sub(r'\-\d+$', '', word)
    
#     # Remove special patterns
#     word = re.sub(r'\]\,\d+\-\d+$', '', word)  # Handling "num],000-5,000"
#     if re.search(r'\.\d+(\,\d+)+$', word):
#         return ''
    
#     # Remove special characters
#     word = re.sub(r'[^a-zA-Z0-9-]', '', word)
#     word = word.strip('.,!;:()[]{}"\'')
    
#     return word

# def preprocess_text_for_analysis(text):
#     """
#     Process a text for analysis: tokenize, remove stopwords and punctuation.

#     Parameters:
#     - text: A string.

#     Returns:
#     A list of tokens.
#     """

#     # Convert to lowercase
#     text = text.lower()

#     # Tokenization
#     # tokens = text.split() # Replace with Spacy tokenization
#     tokens = tokenize_with_spacy(text)

#     # Remove stopwords (using SpaCy's stopword list)
#     tokens = [token for token in tokens if token not in STOP_WORDS]

#     # Remove punctuation from each token
#     translator = str.maketrans('', '', string.punctuation)
#     tokens = [token.translate(translator) for token in tokens]

#     # Preprocess word by word
#     tokens = [process_word(word) for word in tokens]

#     # Remove any tokens with length less than 2
#     tokens = [token for token in tokens if len(token) >= 2]

#     return tokens


# def distinct_words_in_corpus(corpus_tokenized):
#     """
#     Compute the number of distinct words in a corpus.

#     Parameters:
#     - corpus: A list of documents.

#     Returns:
#     The number of distinct words in the corpus.
#     """
#     all_tokens = []
#     for document in corpus_tokenized:
#         # tokens = preprocess_text_for_analysis(document)
#         all_tokens.extend(tokens)
#     counter = Counter(all_tokens)
#     return len(counter.keys())

def distinct_words_in_corpus(corpus_tokenized):
    """
    Compute the number of distinct words in a corpus.

    Parameters:
    - corpus: A list of documents.

    Returns:
    The number of distinct words in the corpus.
    """
    all_tokens = []
    for document in corpus_tokenized:
        # tokens = preprocess_text_for_analysis(document)
        all_tokens.extend(document)
    # Flatten the list of lists
    all_tokens = [token for sublist in all_tokens for token in sublist]
    counter = Counter(all_tokens)
    return len(counter.keys())

# def ngram_frequency_in_corpus(corpus, n=1):
#     """
#     Calculate the most number of occurrences of a given word in a corpus.

#     Parameters:
#     - corpus: A list of documents.
#     - word: The word to count.

#     Returns:
#     The number of occurrences of the word in the corpus.
#     """
#     all_tokens = []
#     for document in corpus:
#         tokens = preprocess_text_for_analysis(document)
#         if n==2:
#             tokens = list(bigrams(tokens))
#         elif n==3:
#             tokens = list(trigrams(tokens))
#         all_tokens.extend(tokens)
#     counter = Counter(all_tokens)
#     return counter

def ngram_frequency_in_corpus(corpus, n=1):
    """
    Calculate the most number of occurrences of a given word in a corpus.

    Parameters:
    - corpus: A list of documents.
    - word: The word to count.

    Returns:
    The number of occurrences of the word in the corpus.
    """
    all_tokens = []
    for document in corpus:
        # flatten the list of lists
        tokens = [token for sublist in document for token in sublist]
        if n==2:
            tokens = list(bigrams(tokens))
        elif n==3:
            tokens = list(trigrams(tokens))
        all_tokens.extend(tokens)
    counter = Counter(all_tokens)
    return counter

# def doc_frequency_in_corpus(corpus):
#     """
#     Calculate the number of documents in which each token (or n-gram) appears in the corpus.

#     Parameters:
#     - corpus: A list of documents.

#     Returns:
#     A Counter object where keys are tokens) and values are the number of documents in which they appear.
#     """
#     doc_frequencies = Counter()

#     for document in corpus:
#         tokens = preprocess_text_for_analysis(document)
#         unique_tokens = set(tokens)  # get unique tokens (or n-grams) for this document
#         doc_frequencies.update(unique_tokens)  # update the global counter with unique tokens

#     return doc_frequencies

def doc_frequency_in_corpus(corpus):
    """
    Calculate the number of documents in which each token (or n-gram) appears in the corpus.

    Parameters:
    - corpus: A list of documents, where each document is a list of sentences,
              and each sentence is a list of words.

    Returns:
    A Counter object where keys are tokens (or n-grams) and values are the number 
    of documents in which they appear.
    """
    doc_frequencies = Counter()

    for document in corpus:
        # Flatten the list of sentences into a single list of words
        flattened_document = [word for sentence in document for word in sentence]
        
        # Get unique tokens (or n-grams) for this document
        unique_tokens = set(flattened_document)
        
        # Update the global counter with unique tokens
        doc_frequencies.update(unique_tokens)

    return doc_frequencies



def darken_color(color, factor=0.5):
    """Darken a color."""
    r, g, b = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

def create_wordclouds(data_dict, method="frequencies", font_path="fonts/Roboto-Regular.ttf", colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
    """
    Create word clouds for each corpus.
    
    Parameters:
    - data_dict: A dictionary with corpus names as keys and the corresponding data as values.
                 If method is 'frequencies', the data should be word frequencies.
                 If method is 'tfidf', the data should be TF-IDF scores.
    - method: A string, either "frequencies" or "tfidf" indicating the word cloud generation method.
    - font_path: The path to the font file to be used for the word clouds.
    - colors: A list of colors to be used for the word clouds.
    """

    # For each corpus, generate and display a word cloud
    for idx, (corpus_name, data) in enumerate(data_dict.items()):

        # Darken the selected color
        darkened_color = darken_color(colors[idx % len(colors)])
        
        # Create a custom colormap based on the darkened color and one of the provided colors
        custom_cmap = LinearSegmentedColormap.from_list("custom", ["white", darkened_color], N=256)

        # Prepare the word cloud input based on the method
        if method == "frequencies":

            # Convert the list of bigram lists to a dictionary of space-separated bigram strings
            ngrams_as_strings = {' '.join(ngram[0]): ngram[1] for ngram in data}

            # Create a word cloud object using the custom colormap
            wc = WordCloud(width=1000, height=600, background_color='white', colormap=custom_cmap, max_words=100, font_path=font_path).generate_from_frequencies(ngrams_as_strings)

            
        elif method == "tfidf":
            wordcloud_input = {word: score for word, score in data.items()}
            # Create a word cloud object using the custom colormap
            wc = WordCloud(width=1000, height=600, background_color='white', colormap=custom_cmap, max_words=100, font_path=font_path).generate_from_frequencies(wordcloud_input)
        else:
            raise ValueError(f"Unsupported method: {method}")


        # Display the word cloud
        plt.figure(figsize=(12, 8))
        plt.imshow(wc, interpolation='bilinear')

        # Add title with the corpus name
        plt.title(corpus_name, fontsize=20, fontweight='bold')

        # Show the word cloud with a frame
        ax = plt.gca()
        [spine.set_visible(True) for spine in ax.spines.values()]

        plt.axis('on')
        plt.show()

def named_entity_recognition(corpus, sample_size=100):
    """
    Perform Named Entity Recognition on a given corpus using SpaCy.

    Parameters:
    - corpus: A list of strings, each string being a document or chunk of text.
    - sample_size: An integer indicating the number of documents to sample from the corpus.

    Returns:
    - A Counter object containing entities and their counts.
    """
    all_entities = []

    # Check if sample_size is less than or equal to the length of the corpus
    # If not, set sample_size to the length of the corpus
    if sample_size > len(corpus):
        sample_size = len(corpus)

    # Randomly sample the desired number of documents from the corpus
    sampled_texts = random.sample(corpus, sample_size)

    # Process each sampled document or chunk of text
    for doc_text in sampled_texts:
        if not isinstance(doc_text, str):
            print(f"Unexpected data type {type(doc_text)} with value {doc_text}")
            continue
        doc = nlp(doc_text)
        entities = [ent.text for ent in doc.ents]
        all_entities.extend(entities)

    # Return the entities and their counts
    return Counter(all_entities)


def ngram_containing_word(ngram_freqs, word):
    """
    Display every n-gram and its frequency in the dictionary where one of the words contains a specific search term, 
    in descending order.

    Parameters:
    - ngram_freqs: Dictionary containing n-gram frequencies by corpus.
    - word: The search term to look for within n-gram words.

    Returns:
    - None (but prints the results)
    """

    ngrams_per_corpus = {}

    for corpus_name, freqs in ngram_freqs.items():
        
        # Filter n-grams where one of the words contains the search term
        matched_ngrams = {ngram: count for ngram, count in freqs.items() 
                          if any(word in w for w in ngram)}
        
        if not matched_ngrams:
            print(f"No n-grams found containing the term '{word}'.")
        else:
            # Sort matched n-grams in descending order based on frequency
            sorted_ngrams = sorted(matched_ngrams.items(), key=lambda x: x[1], reverse=True)
            ngrams_per_corpus[corpus_name] = sorted_ngrams
        
    return ngrams_per_corpus

#####
# 2. Utility Functions for Preprocessing
#####

def is_english_word(word):
    """
    Check if the word is recognized as an English word using WordNet and NLTK's word list.
    """
    # Remove punctuation
    word = ''.join(ch for ch in word if ch not in string.punctuation).lower()

    # Handle contractions by splitting the word
    parts = word.split("'")  # E.g., "don't" will become ["don", "t"]

    # Check each part of the word
    for part in parts:
        if bool(wordnet.synsets(part)) or part in english_words:
            return True

    return False


def filter_english_sentences(text, threshold=0.6):
    """
    Segments the input text into sentences and returns only those sentences that are likely in English.

    Parameters:
    - text: A string containing the input text.
    - threshold: Proportion of recognized English words needed in a sentence to be considered English.

    Returns:
    A string containing only the sentences that are likely in English.
    """
    doc = nlp(text)
    english_sentences = []
    non_english_sentences = []

    # For each sentence in the document
    for sentence in doc.sents:

        # Get alphabetic tokens/words from the sentence
        tokens = [token.text for token in sentence if token.is_alpha]

        # Remove stopwords (using SpaCy's stopword list)
        tokens = [token for token in tokens if token not in STOP_WORDS]

        # Remove punctuation from each token
        translator = str.maketrans('', '', string.punctuation)
        tokens = [token.translate(translator) for token in tokens]

        # Remove any tokens with length less than 2
        tokens = [token for token in tokens if len(token) >= 2]

        # If lenght is 0, skip
        if len(tokens) == 0:
            continue

        # Count the number of recognized English words
        english_word_count = sum(1 for word in tokens if is_english_word(word))
        
        # Check if a sufficient proportion of the sentence is recognized English words
        proportion_english = english_word_count / len(tokens)
        if proportion_english >= threshold:
            english_sentences.append(sentence.text)
        else:
            non_english_sentences.append(sentence.text)
    
    return ' '.join(english_sentences), ' '.join(non_english_sentences)


def store_document(cleaned_text, junk_text, base_dir, subdir, filename):
    """
    Store the processed document to disk with both cleaned and junk texts.

    Parameters:
    - cleaned_text: The cleaned text of the document.
    - junk_text: The junk text of the document.
    - base_dir: The base directory for storing.
    - subdir: The subdirectory for the corpus type.
    - filename: The filename of the original document.
    """
    if not os.path.exists(os.path.join(base_dir, subdir)):
        os.makedirs(os.path.join(base_dir, subdir))
    with open(os.path.join(base_dir, subdir, filename), 'w') as file:
        data = {
            'cleaned_text': cleaned_text,
            'junk': junk_text
        }
        json.dump(data, file)

def document_processed(base_dir, subdir, filename):
    """
    Check if a document has been processed.

    Returns:
    True if the document exists, else False.
    """
    return os.path.exists(os.path.join(base_dir, subdir, filename))

def load_and_print_junk(base_dir, subdirs):
    """
    Load and print all sentences considered junk from the processed documents.

    Parameters:
    - base_dir: The base directory where the preprocessed documents are stored.
    - subdirs: A list of subdirectories inside the base directory.
    """
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                junk_text = data.get('junk', '')
                if junk_text:
                    print(f"Junk from {filename} in {subdir}:")
                    print(junk_text)
                    print('-' * 40)  # separator for better readability


def replace_urls(text):
    url_pattern = re.compile(
        r'(?:(http[s]?://)?(?:www\.)?)([a-zA-Z0-9.-]+\.(?:com|net|org|edu|gov|de|[a-z]{2}))(?:[^\s]*)\b'
    )
    urls_found = [match.group() for match in re.finditer(url_pattern, text)]
    for url in urls_found:
        text = text.replace(url, 'url_token', 1)
    return text, urls_found

def replace_numbers(text):
    # To detect years from 1900-2099
    year_pattern = re.compile(r'\b(?:19|20)\d{2}\b')
    years_found = re.findall(year_pattern, text)
    
    # Replace years
    text = re.sub(year_pattern, 'year_token', text)
    
    # Now detect numbers, excluding the years
    # Preceded by whitespace and followed by whitespace, comma, full stop, or other punctuation
    number_pattern = re.compile(r'(?<=\s)\d+(?=\s|,|\.|!|\?|;|:|-)')
    numbers_found = [num for num in re.findall(number_pattern, text)]
    
    # Replace numbers
    text = re.sub(number_pattern, 'num_token', text)
    
    return text, years_found, numbers_found

def split_text_into_chunks(text, max_length=MAX_TEXT_LENGTH):
    """Split the text into chunks that are smaller than the max_length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def has_letters(input_string):
    return any(char.isalpha() for char in input_string)

def process_sentence(sent, rare_words, replacements):

    cleaned_sentence = []
    for token in sent:
        # Handle tokens that start with one of the special strings but are longer
        if any(token.text.startswith(key) for key in replacements.values()):
            cleaned_sentence.append(token.text.split('_')[0].lower() + '_token')
        elif not token.is_punct and not token.is_space and not token.is_stop:
            # Additional check for tokens with a full stop
            if "." in token.text:
                parts = token.text.split(".")
                for part in parts:
                    # Only process non-empty parts
                    if part:
                        lemma = nlp(part)[0].lemma_
                        if has_letters(lemma) and len(lemma) > 2:  # Check if token has at least 3 characters
                            if lemma not in rare_words:
                                cleaned_sentence.append(lemma)
                            else:
                                cleaned_sentence.append("rare_token")
            else:
                lemma = token.lemma_
                if has_letters(lemma) and len(lemma) > 2:  # Check if token has at least 3 characters
                    if lemma not in rare_words:
                        cleaned_sentence.append(lemma)
                    else:
                        cleaned_sentence.append("rare_token")

    return cleaned_sentence

def process_corpus(corpus, replacements):

    combined_text = ' '.join(corpus)
    for original, replacement in replacements.items():
        combined_text = combined_text.replace(original, replacement)
    combined_text = combined_text.lower()

    return combined_text


def preprocess_for_word2vec(corpus, word_freqs, threshold=100):
    """
    Preprocesses a corpus of text for Word2Vec model.
    
    Parameters:
    - corpus: List of individual text strings representing documents.
    - word_freqs: Dictionary of word frequencies for the corpus.
    - threshold: Minimum frequency for a word to be retained. Default is 100.

    Returns:
    - A list of lists (sentences) where each sentence is a list of individual words.
    """
    
    # Temporary replacements for special tokens
    replacements = {'[NUM]': 'num_token', '[YEAR]': 'year_token', '[URL]': 'url_token'}
    
    # Combine the corpus into one large text, replace special tokens, and lowercase
    combined_text = process_corpus(corpus, replacements)

    # Split the combined text into manageable chunks (due to Spacy maximum text length)
    chunks = split_text_into_chunks(combined_text)

    # Create a set of rare words for efficient lookup
    rare_words = set([word for word, count in word_freqs.items() if count < threshold])
    
    # For each sentence, remove punctuation, trailing spaces, stopwords, short tokens, and replace rare tokens with rare_token
    cleaned_sentences = []
    for i, chunk in enumerate(chunks):

        # If i is a multiple of 1000, print the progress
        if i % 50 == 0:
            print(f"Chunk {i} of {len(chunks)}")

        doc = nlp(chunk)
        for sent in doc.sents:
            cleaned_sentence = process_sentence(sent, rare_words, replacements)
            if cleaned_sentence:  # Only add non-empty sentences
                cleaned_sentences.append(cleaned_sentence)
    
    return cleaned_sentences

def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())


#####
# 3. Utility Functions for Creating Models and Evaluation
#####

def process_bigram(word):

    # Remove trailing spaces in word
    word = word.strip()

    # Replace remaining spaces in word with underscore
    word = word.replace(' ', '_')

    return word
    

def find_closest_words(model1, model1_name, model2, model2_name, word, top_n=10):

    # Remove trailing spaces in word
    word = process_bigram(word)

    # Initialize lists to hold the similar words from each model
    similar_words_model1 = []
    similar_words_model2 = []

    # Check and find similar words for the first model
    if word in model1.wv.key_to_index:
        similar_words_model1 = model1.wv.most_similar(word, topn=top_n)
    else:
        print(f"'{word}' not found in the first model.")

    # Check and find similar words for the second model
    if word in model2.wv.key_to_index:
        similar_words_model2 = model2.wv.most_similar(word, topn=top_n)
    else:
        print(f"'{word}' not found in the second model.")

    # Create a DataFrame to hold the results
    df = pd.DataFrame({
        f'{model1_name} - Word': [word.replace("_"," ") for word, _ in similar_words_model1],
        f'{model1_name} - Score': [score for _, score in similar_words_model1],
        f'{model2_name} - Word': [word.replace("_"," ") for word, _ in similar_words_model2],
        f'{model2_name} - Score': [score for _, score in similar_words_model2]
    })

    return df

def compare_words_in_models(model1, model1_name, model2, model2_name, word1, word2):

    # Process words in case they are bigrams:
    word1 = process_bigram(word1)
    word2 = process_bigram(word2)

    # Initialize dictionary
    results = {}

    # Function to compute cosine similarity within a model
    def compute_similarity(model, model_name, word1, word2):
        # Check if both words are in the model's vocabulary
        if word1 in model.wv.key_to_index and word2 in model.wv.key_to_index:
            # Compute cosine similarity (ranges from -1 to 1; higher means more similar)
            cos_similarity = model.wv.similarity(word1, word2)
            results[model_name] = cos_similarity
        else:
            print(f"One or both words not found in {model_name}.")

    # Compute similarity for each model
    compute_similarity(model1, model1_name, word1, word2)
    compute_similarity(model2, model2_name, word1, word2)

    return results

def print_similarity_results(results, word1, word2):
 
    print(f"Similarity between '{word1}' and '{word2}':")
    for model_name, similarity in results.items():
        if similarity is not None:
            print(f"In {model_name}: {similarity:.4f}")
        else:
            print(f"Could not compute similarity in {model_name}.")

def plot_wv_pca(model1, model1_name, model2, model2_name, words, show_distance=True):

    # Process words in case they are bigrams
    words = [process_bigram(word) for word in words]

    # For each model
    for h,model in enumerate([model1, model2]):

        # Get the vectors for the words
        word_vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]

        # Apply PCA to reduce dimensions to 2
        pca = PCA(n_components=2)
        result = pca.fit_transform(word_vectors)

        # Create a scatter plot of the projection
        plt.figure(figsize=(12, 10))

        if show_distance:
            # Annotate points and draw lines
            for i, word in enumerate(words):
                if word in model.wv.key_to_index:
                    for j in range(i+1, len(words)):
                        if words[j] in model.wv.key_to_index:
                            # Draw line
                            plt.plot([result[i, 0], result[j, 0]], [result[i, 1], result[j, 1]], 'k-', alpha=0.2)
                            # Calculate and plot cosine similarity
                            cos_sim = model.wv.similarity(words[i],words[j])
                            mid_point = [(result[i, 0] + result[j, 0])/2, (result[i, 1] + result[j, 1])/2]
                            plt.text(mid_point[0], mid_point[1], f"{cos_sim:.2f}", color='lightblue', alpha=0.9)
                model_name = model1_name if h==0 else model2_name
            plt.title(f"Word Vector Space for {model_name}")

        # Create a scatter plot of the projection
        plt.scatter(result[:, 0], result[:, 1])
        offset = 0.2
        for i, word in enumerate(words):
            if word in model.wv.key_to_index:
                plt.annotate(word.replace("_"," "), xy=(result[i, 0]+offset, result[i, 1]+offset))


        plt.show()

def plot_wv_heatmap(model1, model1_name, model2, model2_name, words):
    for m,model in enumerate([model1, model2]):

        # Process words in case they are bigrams
        words = [process_bigram(word) for word in words]

        # Check if the words are in the vocabulary and get their vectors
        word_vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]

        # Calculate the pairwise cosine distances
        n = len(word_vectors)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distance_matrix[i, j] = model.wv.similarity(words[i], words[j])
                distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric matrix

        # Create a triangular mask for the heatmap
        mask = np.triu(np.ones_like(distance_matrix, dtype=bool))

        # Create ticklabels by removing underscores
        words_readable = [word.replace("_"," ") for word in words]
        xticklabels = words_readable[:-1]
        yticklabels = words_readable
        yticklabels[0] = yticklabels[0].replace(words_readable[0],"")

        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        plt.title(f'Similarity Heatmap for {model1_name if m==0 else model2_name} Word2Vec model')
        sns.heatmap(distance_matrix, mask=mask, cmap='viridis', annot=True, xticklabels=xticklabels, yticklabels=yticklabels)
        plt.show()

def similiarty_to_vaccine(word, model):
    """
    Calculates the similarity between a word and the word "vaccine" in the model. If the word is not in the vocabulary, returns None.
    """
    bigram = process_bigram(word)

    if bigram in model.wv.key_to_index:
        similarity = model.wv.similarity(bigram,"vaccine")
        return similarity
    else:
        return None
    

#####
# 4. Utility Functions for Topic Modeling
#####
    
def load_corpora_to_df(base_dir, subdirs, filter=True):
    """
    Load corpora from specified directories, filtering out documents with less than 100 words,
    and returns a DataFrame with columns for document text, corpus label, and filename.

    Parameters:
    - base_dir: The base directory where the corpora are stored.
    - subdirs: A list of subdirectories inside the base directory.
    - filter: Boolean to indicate whether to filter out documents with less than 100 words.

    Returns:
    - DataFrame with columns 'Document' for the document texts, 'Corpus' for the subdirectory names, and 'Filename' for the names of the files.
    - Dictionary with the count of filtered documents per corpus if filter is True.
    """
    documents = []  # Initialize a list to store (document, corpus, filename) tuples
    filtered_counts = {}  # Dictionary to count filtered documents

    # Iterate through each subdirectory and load documents
    for subdir in subdirs:
        filtered_count = 0  # Reset filtered count for each subdir

        for filename in os.listdir(os.path.join(base_dir, subdir)):
            if filename.endswith('.json'):
                with open(os.path.join(base_dir, subdir, filename), 'r') as file:
                    data = json.load(file)
                    text = data.get('cleaned_text', '')  # Use get to avoid KeyError if 'cleaned_text' does not exist
                    if text:
                        word_count = len(text.split())
                        if filter and word_count >= 100:
                            documents.append((text, subdir, filename))  # Include filename in the tuple
                        elif not filter:
                            documents.append((text, subdir, filename))  # Include filename in the tuple
                        else:
                            filtered_count += 1

        # Update the filtered counts
        filtered_counts[subdir] = filtered_count

    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(documents, columns=['Document', 'Corpus', 'Filename'])  # Add 'Filename' to the columns

    return df, filtered_counts
