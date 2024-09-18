from flask import Flask, request, jsonify
from pyngrok import ngrok
import pandas as pd
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish

app = Flask(__name__)

# Load your data from Excel
file_path = '/Users/the_lost_kid/Desktop/sihp2/backend/dataset/output_file1.xlsx'

try:
    df = pd.read_excel(file_path)
    print(f"Loaded file: {file_path}")
    df.columns = ['Title-Code', 'Title Name', 'Hindi Title', 'Register Serial No.', 'Regn No.', 'Owner Name', 'State', 'Publication City/District', 'Language Code']
except Exception as e:
    print(f"Error loading file: {e}")
    df = None

# Preprocessing functions
common_prefixes_suffixes = ["The", "India", "Samachar", "News", "Daily", "Dainik", "Saaptahik", "Maasik"]
disallowed_words = ["police", "crime", "corruption", "army", "cbi", "cid"]

def preprocess(title):
    title = title.lower()
    for word in common_prefixes_suffixes:
        title = title.replace(word.lower(), "")
    return title.strip()

def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^a-z\s]', '', title)
    title = ' '.join([word for word in title.split() if word not in disallowed_words])
    return title.strip()

if df is not None:
    df['Cleaned Title Name'] = df['Title Name'].apply(clean_title)

# Similarity calculation function
def calculate_similarity(input_title, df):
    cleaned_input = clean_title(input_title)

    # Levenshtein Similarity
    df['Levenshtein Similarity'] = df['Cleaned Title Name'].apply(lambda x: fuzz.ratio(cleaned_input, x))

    # Cosine Similarity
    vectorizer = TfidfVectorizer().fit_transform(df['Cleaned Title Name'].tolist() + [cleaned_input])
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])[0]
    df['Cosine Similarity'] = cosine_sim

    # Phonetic Similarity
    input_soundex = jellyfish.soundex(cleaned_input)
    df['Phonetic Similarity'] = df['Cleaned Title Name'].apply(lambda x: jellyfish.soundex(x) == input_soundex)

    # Apply similarity threshold logic (to match second script)
    similarity_threshold = 80
    similar_titles = df[(df['Levenshtein Similarity'] > similarity_threshold) | (df['Phonetic Similarity'])]

    result_columns = ['Title Name', 'Levenshtein Similarity', 'Cosine Similarity', 'Phonetic Similarity']
    if 'Hindi Title' in df.columns:
        result_columns.insert(1, 'Hindi Title')

    return similar_titles[result_columns]

# Flask route to check title similarity
@app.route('/check-title', methods=['POST'])
def check_title_similarity():
    data = request.get_json()
    input_title = data.get('title', '').strip()

    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    if not input_title:
        return jsonify({"error": "No title provided"}), 400

    result = calculate_similarity(input_title, df)

    if not result.empty:
        # Top 5 Similar Titles
        top_similar_titles = result.nlargest(5, 'Levenshtein Similarity')

        # Acceptable Unique Titles (titles with a similarity score not exceeding the threshold)
        similarity_threshold = 80
        acceptable_unique_titles = df[~df['Title Name'].isin(result['Title Name'])]
        acceptable_unique_titles = acceptable_unique_titles[
            (acceptable_unique_titles['Levenshtein Similarity'] <= similarity_threshold) &
            (~acceptable_unique_titles['Phonetic Similarity'])
        ]

        return jsonify({
            "message": "The input title is too similar to existing titles and cannot be accepted.",
            "top_similar_titles": top_similar_titles[['Title Name']].drop_duplicates().to_dict(orient='records'),
            "acceptable_unique_titles": acceptable_unique_titles[['Title Name']].head(5).drop_duplicates().to_dict(orient='records')
        }), 400
    else:
        return jsonify({
            "message": "The input title does not have significant similarity to existing titles and can be accepted.",
            "acceptable_unique_titles": result[['Title Name']].drop_duplicates().to_dict(orient='records')
        }), 200

# Expose the Flask app using pyngrok
if __name__ == '__main__':
    # Open an ngrok tunnel to the HTTP server
    public_url = ngrok.connect(5000)
    print(f" * Ngrok Tunnel URL: {public_url}")
    app.run(port=5000)