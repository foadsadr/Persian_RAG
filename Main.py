import numpy as np
import pandas as pd
from hazm import stopwords_list, Normalizer, WordTokenizer, SentenceTokenizer, Lemmatizer, sent_tokenize, word_tokenize
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from rake_nltk import Rake
import stanza
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os 
nlp = None
api_key=os.environ.get('OPENROUTER_API_KEY')
# if not api_key:
#     raise ValueError("API key not found. Set OPENROUTER_API_KEY environment variable.")
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=api_key,
# )
def setup_stanza():
    print("Setting up Stanza pipeline for Persian ('fa')...")
    stanza.download('fa')
    pipeline = stanza.Pipeline('fa')  
    print("Stanza setup complete.")
    return pipeline
    #----class definition----
class PersianRAKE(Rake):
    def _tokenize_text_to_sentences(self, text: str):
        return word_tokenize(text)
    def _tokenize_sentence_to_words(self, sentence: str):
        return word_tokenize(sentence)
    # --- Helper Functions (Mostly from Colab/Original) ---
    def read_from_docx(doc_object):
        fullText=''
        for para in doc_object.paragraphs:
            fullText += para.text + ' '
        return fullText
    def split_into_overlapping_chunks(sentences, max_chunk_size=1000, overlap_size=200):   
        """Splits a list of sentences into overlapping text chunks."""
        chunks = []
        current_chunk = ""
        current_chunk_size = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_chunk_size + sentence_length > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Careful slicing to get overlap correctly
                overlap_start_index = max(0, len(current_chunk) - overlap_size)
                # Find the nearest space to start overlap cleanly (optional but good)
                space_index = current_chunk.rfind(" ", overlap_start_index - 20, overlap_start_index + 20) # Search around target
                if space_index != -1:
                    overlap_start_index = space_index + 1

                overlap_buffer = current_chunk[overlap_start_index:].strip()
                current_chunk = overlap_buffer + (" " if overlap_buffer else "")
                current_chunk_size = len(current_chunk)

            current_chunk += sentence + " "
            current_chunk_size += sentence_length + 1

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
def preprocess_text_1(text):
    """Basic cleaning: remove punctuation (keep alphanumeric, underscore, space), normalize spaces."""
    text = re.sub(r'[^\w\s]', '', text) # Keep word chars (incl. Persian), underscore, whitespace
    text = re.sub(r'\s+', ' ', str(text)).strip() # Normalize whitespace and strip ends
    return text
def preprocess_text_2(text):
    """Advanced Persian preprocessing: remove bracketed text, tokenize, remove stopwords, lemmatize."""
    text = re.sub(r'(\(.*?\))|(\[.*?\])', '', str(text)) # Remove () and [] content
    text = re.sub(r'\s+', ' ', str(text)).strip() # Normalize spaces again

    word_tokenizer = WordTokenizer()
    words = word_tokenizer.tokenize(text)

    # Consider edge case: empty text after preprocessing
    if not words:
        return ""
def check_spelling(main_text):
    """Checks spelling using LanguageTool API (WARNING: Configured for en-US)."""
    # Consider removing this or finding a Persian-compatible tool
    print("--- Running Spell Check (WARNING: Using en-US) ---")
    endpoint = "https://api.languagetool.org/v2/check"
    data = {
        "text": main_text,
        "language": "en-US", # <<< PROBLEM: Incorrect for Persian
    }
    try:
        response = requests.post(endpoint, data=data, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes
        json_response = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Spell check API request failed: {e}")
        return main_text # Return original text on error
    updated_text = main_text
    # Process matches carefully to avoid incorrect replacements on repeated strings
    matches = sorted(json_response.get("matches", []), key=lambda x: x['offset'], reverse=True)
    for match in matches:
        if match["replacements"]:
            replacement = match["replacements"][0]["value"]
            offset = match["offset"]
            length = match["length"]
            updated_text = updated_text[:offset] + replacement + updated_text[offset+length:]

    print(f"Original Query: {main_text}")
    print(f"Spell-checked Query (en-US): {updated_text}")
    print("--- Spell Check Complete ---")
    return updated_text
def phrase_search(sentence):
    """Finds simple noun/adjective phrases using Stanza POS tagging."""
    global nlp # Uses the global Stanza pipeline
    if nlp is None:
        print("Error: Stanza pipeline (nlp) not initialized.")
        return []
    doc = nlp(sentence)
    phrases = []
    # Simplified phrase extraction logic (as in Colab)
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['NOUN', 'ADJ']:
                # Look for direct dependents that are NOUN or ADJ
                dependent_phrase_parts = [other_word.text for other_word in sent.words
                                          if other_word.head == word.id and other_word.upos in ['NOUN', 'ADJ']]
                if dependent_phrase_parts:
                    # Basic phrase: head + dependents
                    phrase = word.text + " " + " ".join(dependent_phrase_parts)
                    phrases.append(phrase)
                    # Consider also phrases formed by adjectives modifying nouns (adj -> noun)
                    # This logic might need refinement based on desired phrase types.
                # Look for ADJ modifying this word if it's a NOUN
                if word.upos == 'NOUN':
                     modifying_adjs = [other_word.text for other_word in sent.words
                                       if other_word.head == word.id and other_word.upos == 'ADJ']
                     if modifying_adjs:
                         phrase = " ".join(modifying_adjs) + " " + word.text
                         phrases.append(phrase)


    # Filter for multi-word phrases and remove duplicates
    unique_phrases = list(set([p for p in phrases if " " in p]))
    return unique_phrases
def english_to_persian_number(number_str):
    """Converts ASCII digits string to Persian digits string."""
    english_to_persian = {"0": "۰","1": "۱","2": "۲","3": "۳","4": "۴","5": "۵","6": "۶","7": "۷","8": "۸","9": "۹"}
    return "".join([english_to_persian.get(digit, digit) for digit in number_str]) # Handle non-digits gracefully

def persian_words_to_number(sentence):
    """Converts Persian number words in a sentence to Persian digits."""
    word_to_number = {
        "صفر": 0, "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "نه": 9,
        "ده": 10, "یازده": 11, "دوازده": 12, "سیزده": 13, "چهارده": 14, "پانزده": 15, "شانزده": 16, "هفده": 17, "هجده": 18, "نوزده": 19,
        "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90,
        "صد": 100, "یکصد": 100, "دویست": 200, "سیصد": 300, "چهارصد": 400, "پانصد": 500, "ششصد": 600, "هفتصد": 700, "هشتصد": 800, "نهصد": 900,
        "هزار": 1000, # Basic handling, doesn't combine (e.g., هزار و دویست)
    }
    words = sentence.split(' ')
    result = []
    temp_number_words = []
    current_number = 0
    has_va = False # Simple flag to handle "و" like "بیست و یک"

    for word in words:
        original_word = word
        is_ordinal = False
        # Handle ordinals like 'هفتم'
        if word.endswith('م') and word[:-1] in word_to_number:
            word = word[:-1]
            is_ordinal = True

        if word == 'و':
             has_va = True
             # Append 'و' if it's not between numbers being combined
             if not temp_number_words:
                  result.append(original_word)
             continue # Process next word

        if word in word_to_number:
            value = word_to_number[word]
            if not has_va and temp_number_words: # New number sequence starts (e.g., "پنج ... ده")
                 # Finalize previous number
                 english_number_str = str(current_number)
                 persian_number_str = english_to_persian_number(english_number_str)
                 result.append(persian_number_str)
                 temp_number_words = []
                 current_number = 0

            # Basic combination logic (e.g., for بیست و یک, سی و ...)
            # More complex logic needed for hundreds/thousands + tens/ones
            if value < 100 and current_number >= 100: # e.g., صد و بیست
                 current_number += value
            elif value >= 100 and current_number > 0 and current_number < 1000: # e.g., دویست هزار
                current_number *= value # Simplistic thousand combination
            elif value >= 20 and current_number > 0 and current_number < 10: # e.g., بیست و یک
                current_number += value
            else: # Default: add (or start if first number)
                 current_number = value if not temp_number_words else current_number + value

            temp_number_words.append(word)
            has_va = False # Reset 'و' flag after processing number
        else:
            # Word is not a number, finalize any pending number
            if temp_number_words:
                english_number_str = str(current_number)
                persian_number_str = english_to_persian_number(english_number_str)
                result.append(persian_number_str)
                temp_number_words = []
                current_number = 0
            result.append(original_word) # Append the non-number word
            has_va = False # Reset 'و' flag

    # Append any remaining number at the end
    if temp_number_words:
        english_number_str = str(current_number)
        persian_number_str = english_to_persian_number(english_number_str)
        result.append(persian_number_str)

    return ' '.join(result)
def preprocess_phrases(text, phrases):
    """Replaces spaces in identified phrases with underscores."""
    processed_text = text
    # Process longer phrases first to avoid partial replacements
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    for phrase in sorted_phrases:
        processed_text = processed_text.replace(phrase, phrase.replace(" ", "_"))
    return processed_text

def extract_persian_numbers(text):
    """Extracts sequences of Persian digits from text using regex."""
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    return re.findall(f"[{persian_digits}]+", text)

def calculate_tf_idf_similarity(docs, query_list):
    """Calculates TF-IDF cosine similarity between docs and a query."""
    # Ensure query is a list
    if isinstance(query_list, str):
        query_list = [query_list]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3)) # Using ngrams 1-3 as in Colab
    try:
         # Handle case where docs might be empty after preprocessing
         if not docs:
              print("Warning: Empty document list provided for TF-IDF.")
              # Return zero similarity for the expected shape (len(query_list), 0)
              return np.zeros((len(query_list), 0))
         tfidf_matrix = vectorizer.fit_transform(docs)
         # Handle case where query might be empty
         if not query_list or all(not q for q in query_list):
             print("Warning: Empty query provided for TF-IDF.")
             # Return zero similarity for shape (0, num_docs) or handle as appropriate
             return np.zeros((len(query_list), tfidf_matrix.shape[0]))

         phrase_vector = vectorizer.transform(query_list)
         return cosine_similarity(phrase_vector, tfidf_matrix)
    except ValueError as e:
         # Catch potential "empty vocabulary" errors if docs/query are empty after processing
         print(f"TF-IDF Vectorization Error: {e}. Check preprocessed text.")
         # Return zero similarity, adjust shape based on input
         num_docs = len(docs) if docs else 0
         return np.zeros((len(query_list), num_docs))


def calculate_tf(document_numbers):
    """Calculates term frequency (TF) for numbers in lists."""
    tf = []
    for doc in document_numbers:
        tf_dict = defaultdict(int)
        for num in doc:
            tf_dict[num] += 1
        tf.append(dict(tf_dict)) # Convert back to regular dict
    return tf

def calculate_idf(document_numbers, numbers_set):
    """Calculates inverse document frequency (IDF) for numbers."""
    idf = {}
    total_docs = len(document_numbers)
    if total_docs == 0: return {} # Handle empty input

    # Pre-calculate document presence for efficiency
    doc_presence = defaultdict(int)
    for doc in document_numbers:
        for num in set(doc): # Count each number only once per doc for IDF
             if num in numbers_set:
                 doc_presence[num] += 1

    for num in numbers_set:
        doc_count = doc_presence[num]
        # IDF calculation with smoothing
        idf[num] = np.log((total_docs + 1) / (doc_count + 1)) + 1
    return idf

def calculate_tf_idf_for_numbers(document_numbers, query_numbers):
    """Calculates TF-IDF scores for specific query numbers within documents."""
    query_numbers_set = set(query_numbers)
    if not query_numbers_set or not document_numbers:
        return [{} for _ in document_numbers] # Return empty scores if no query numbers or docs

    tf = calculate_tf(document_numbers)
    idf = calculate_idf(document_numbers, query_numbers_set)
    tf_idf_scores = []

    for doc_tf_dict in tf:
        doc_tf_idf = {}
        for num, freq in doc_tf_dict.items():
            if num in idf: # Calculate score only for numbers present in the query set AND having an IDF
                doc_tf_idf[num] = freq * idf[num]
        tf_idf_scores.append(doc_tf_idf)
    return tf_idf_scores

def generate_answer(top_k_chunks, query, api_key):
    """Generates an answer using OpenAI/OpenRouter based on provided chunks and query."""
    if not api_key:
        print("Error: API Key not provided for answer generation.")
        return "API Key missing."
    if not top_k_chunks:
        print("Warning: No relevant chunks found to generate answer.")
        return "متن مرتبطی یافت نشد."
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    context = "\n".join(top_k_chunks)
    prompt = f"{context}\n\nطبق متن های بالا به طور خلاصه(در حد یک پاراگراف) به این سوال جواب بده و اشاره ای به کلمه پاراگراف نکن:{query}\n"

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini", # Updated model name for OpenRouter potentially
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250 # Add max tokens limit
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return f"خطا در ارتباط با سرویس: {e}"
        # --- Main Execution ---
def main():
    """Main function to run the RAG pipeline."""
    global nlp # To assign the initialized pipeline

    # --- Configuration ---
    DOCUMENT_PATH = "C:/Users/f.haghighi/Desktop/Persian_RAG/doc.docx" # <<< CHANGE THIS PATH
    API_KEY_PATH = "C:/Users/f.haghighi/Desktop/Persian_RAG/api_key.csv" # <<< CHANGE THIS PATH (or use environment variables)
    API_KEY_COLUMN = 'api_key' # Column name in the CSV file

    QUERY_COEF = 0.5
    NUMBERS_COEF = 0.3
    PHRASES_COEF = 0.2
    TOP_K = 3

    # --- Setup ---
    try:
        nlp = setup_stanza()
    except Exception as e:
        print(f"Failed to setup Stanza: {e}")
        return # Exit if Stanza fails

    # --- Load Document ---
    try:
        print(f"Loading document from: {DOCUMENT_PATH}")
        doc_obj = docx.Document(DOCUMENT_PATH)
        document_text = read_from_docx(doc_obj)
        print(f"Document loaded successfully ({len(document_text)} characters).")
    except FileNotFoundError:
        print(f"Error: Document file not found at {DOCUMENT_PATH}")
        return
    except Exception as e:
        print(f"Error loading document: {e}")
        return

    # --- Chunking ---
    print("Normalizing and chunking document...")
    normalizer = Normalizer()
    normalized_text = normalizer.normalize(document_text)
    sentence_tokenizer = SentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(normalized_text)
    chunks = split_into_overlapping_chunks(sentences, max_chunk_size=1000, overlap_size=200)
    print(f"Document split into {len(chunks)} chunks.")
    if not chunks:
        print("Error: No chunks created from the document.")
        return
    print("<Chunk 1 Example>")
    print(f"Original Chunk:\n{chunks[0][:200]}...") # Print only beginning

    # --- Preprocessing Chunks ---
    print("Preprocessing chunks...")
    preprocessed1_chunks = [preprocess_text_1(chunk) for chunk in chunks]
    preprocessed2_chunks = [preprocess_text_2(chunk) for chunk in preprocessed1_chunks]
    # Chunks after number conversion (used for TF-IDF base)
    numberic_chunks = [persian_words_to_number(chunk) for chunk in preprocessed2_chunks]
    print(f"Preprocessed Chunk Example:\n{preprocessed2_chunks[0][:200]}...")
    print(f"Numeric Chunk Example:\n{numberic_chunks[0][:200]}...")

    # --- User Query ---
    # query = input("Enter your query in Persian: ") # Or use a hardcoded query for testing
    query = "اصل هفتم قانون اساسی جمهوری اسلامی درباره چیست؟"
    print(f"\nProcessing Query: {query}")

    # --- Query Processing ---
    # Consider removing spell check if not working for Persian
    # query_checked = check_spelling(query) # Optional, likely problematic
    query_checked = query # Skip spell check for now

    processed_query = preprocess_text_2(preprocess_text_1(query_checked))
    query_n = persian_words_to_number(processed_query)
    query_phrases = phrase_search(query_n)
    query_numbers = extract_persian_numbers(query_n) # Fixed to use correct function

    print(f"Processed Query: {processed_query}")
    print(f"Query with Numbers: {query_n}")
    print(f"Extracted Phrases: {query_phrases}")
    print(f"Extracted Numbers: {query_numbers}")

    # --- Calculate Hybrid Similarity ---
    print("\nCalculating hybrid similarity scores...")

    # 1. Query TF-IDF Similarity
    query_bonus = calculate_tf_idf_similarity(numberic_chunks, [query_n]).flatten()
    print(f"Query Bonus Shape: {query_bonus.shape}")


    # 2. Number Bonus
    document_numbers = [extract_persian_numbers(doc) for doc in numberic_chunks]
    tf_idf_scores_for_numbers = calculate_tf_idf_for_numbers(document_numbers, query_numbers)

    number_bonus = np.zeros(len(numberic_chunks))
    if query_numbers: # Only calculate if query has numbers
        for i, doc_scores in enumerate(tf_idf_scores_for_numbers):
            # Sum the TF-IDF scores of query numbers found in this doc
            number_bonus[i] = sum(doc_scores.get(num, 0) for num in query_numbers)

        # Normalize
        max_num_bonus = np.max(number_bonus)
        if max_num_bonus > 0:
            number_bonus = number_bonus / max_num_bonus
    print(f"Number Bonus Shape: {number_bonus.shape}")


    # 3. Phrase Bonus
    phrases_bonus = np.zeros(len(numberic_chunks))
    if query_phrases: # Only calculate if query has phrases
        preprocessed_docs_for_phrases = [preprocess_phrases(doc, query_phrases) for doc in numberic_chunks]
        preprocessed_query_for_phrases = preprocess_phrases(query_n, query_phrases)
        phrases_similarity = calculate_tf_idf_similarity(preprocessed_docs_for_phrases, [preprocessed_query_for_phrases])
        phrases_bonus = phrases_similarity.flatten() # Fixed flatten logic

        # Normalize
        max_phrase_bonus = np.max(phrases_bonus)
        if max_phrase_bonus > 0:
            phrases_bonus = phrases_bonus / max_phrase_bonus
    print(f"Phrase Bonus Shape: {phrases_bonus.shape}")


    # Combine Scores
    hybrid_scores = (query_bonus * QUERY_COEF) + \
                    (number_bonus * NUMBERS_COEF) + \
                    (phrases_bonus * PHRASES_COEF)

    # --- Rank and Select Top K ---
    print("\nRanking chunks...")
    # Ensure shapes match before combining if there were errors
    if not (len(hybrid_scores) == len(chunks)):
         print(f"Error: Score array length ({len(hybrid_scores)}) does not match chunk count ({len(chunks)}). Skipping ranking.")
         return

    indices = np.argsort(-hybrid_scores)[:TOP_K]
    top_k_chunks = [chunks[idx] for idx in indices] # Use original chunks for LLM

    print(f"\n--- Top {TOP_K} Relevant Chunks ---")
    for i, idx in enumerate(indices):
        print(f"Rank {i+1} (Chunk Index: {idx}, Score: {hybrid_scores[idx]:.4f})")
        print(f"{top_k_chunks[i][:300]}...\n") # Print snippet

    # --- Load API Key ---
    api_key = None
    try:
        print(f"Loading API key from: {API_KEY_PATH}")
        api_df = pd.read_csv(API_KEY_PATH)
        if API_KEY_COLUMN in api_df.columns and not api_df.empty:
            api_key = api_df.loc[0, API_KEY_COLUMN]
            print("API Key loaded successfully.")
        else:
            print(f"Error: Column '{API_KEY_COLUMN}' not found or CSV is empty.")
    except FileNotFoundError:
        print(f"Error: API key file not found at {API_KEY_PATH}")
        print("Skipping answer generation.")
    except Exception as e:
        print(f"Error reading API key file: {e}")
        print("Skipping answer generation.")

    # --- Generate Answer ---
    if api_key and top_k_chunks:
        print("\n--- Generating Final Answer ---")
        final_answer = generate_answer(top_k_chunks, query, api_key)
        print("\nFinal Answer:")
        print(final_answer)
    elif not top_k_chunks:
         print("\nSkipping answer generation as no relevant chunks were found.")
    else:
         print("\nSkipping answer generation due to missing API key.")


if __name__ == "__main__":
    main()

# --- END OF FILE Main.py ---
