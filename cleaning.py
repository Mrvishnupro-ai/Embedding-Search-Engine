import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

def remove_email_headers(text):
    lines = text.split('\n')
    clean_lines = []
    header_keywords = ["From:", "Organization:", "Lines:", "Distribution:", "NNTP-Posting-Host:", "Article-I.D.:", "In-Reply-To:"]
    
    for line in lines:
        is_header = False
        for keyword in header_keywords:
            if line.strip().startswith(keyword):
                is_header = True
                break
        if not is_header:
            clean_lines.append(line)
            
    return '\n'.join(clean_lines)

def tokenize_text(text):
    return word_tokenize(text)

def convert_to_lowercase(text):
    return text.lower()

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def stem_text(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def remove_duplicates(texts):
    return list(set(texts))

def correct_spelling(text):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    tokens = word_tokenize(text)
    corrected_tokens = [spell.correction(word) for word in tokens]
    corrected_tokens = [word if word is not None else "" for word in corrected_tokens]
    return ' '.join(corrected_tokens)

def fix_encoding(text):
    try:
        return text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return 'Encoding Error'

def remove_whitespace(text):
    return ' '.join(text.split())

def parallel_process_text(data, cleaning_function, num_workers):
    with Pool(num_workers) as pool:
        cleaned_data = pool.map(cleaning_function, data)
    return cleaned_data

def summarize_long_document(text, ratio=0.2):
    from gensim.summarization import summarize
    try:
        return summarize(text, ratio=ratio)
    except ValueError:
        return text

def remove_emails(text):
    return re.sub(r'\S+@\S+', '', text)

def remove_urls(text):
    return re.sub(r"http\S+|www\.\S+", "", text)

def clean_text_pipeline(text):
    text = fix_encoding(text)
    text = remove_email_headers(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_special_characters(text)
    text = remove_whitespace(text)
    text = convert_to_lowercase(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return ' '.join(tokens)


