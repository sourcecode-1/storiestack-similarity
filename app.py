#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 🔧 Hugging Face cache path
os.environ["HF_HOME"] = "/app/cache"

# 🚀 Initialize Flask
app = Flask(__name__)

# 🧠 Load SentenceTransformer model
print("Loading SentenceTransformer model...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/app/cache')
print("Model loaded successfully.")

# 🧹 Normalize input sentences
def normalize_sentences(sentences):
    return [s.strip().lower() for s in sentences]

# 🏗️ Build FAISS cosine index
def build_index(sentences):
    print("Encoding reference sentences...")
    embeddings = model.encode(sentences)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings

# 🔍 Find the most similar sentence with a threshold
def find_most_similar(index, ref_sentences, input_sentence, threshold=0.4):
    input_embedding = model.encode([input_sentence])
    input_embedding = np.array(input_embedding).astype("float32")
    faiss.normalize_L2(input_embedding)

    distances, indices = index.search(input_embedding, k=1)
    idx = indices[0][0]
    distance = distances[0][0]

    if distance < threshold:
        return input_sentence, None, distance

    return input_sentence, ref_sentences[idx], distance

# 🛠️ Health check
@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Similarity service is running."})

# 🌐 Main API endpoint
@app.route('/find_similar_dynamic', methods=['POST'])
def find_similar_dynamic_route():
    data = request.get_json()

    if not data or "sentences" not in data or "sentence" not in data:
        return jsonify({
            "error": "Please provide both 'sentences' (list) and 'sentence' (query)."
        }), 400

    ref_sentences_raw = data["sentences"]
    input_sentence_raw = data["sentence"]

    if not isinstance(ref_sentences_raw, list) or not isinstance(input_sentence_raw, str):
        return jsonify({
            "error": "'sentences' must be a list and 'sentence' must be a string."
        }), 400

    try:
        ref_sentences = normalize_sentences(ref_sentences_raw)
        input_sentence = input_sentence_raw.strip().lower()

        index, _ = build_index(ref_sentences)
        matched_input, most_similar, distance = find_most_similar(
            index, ref_sentences, input_sentence, threshold=0.4
        )

        if most_similar is None:
            return jsonify({
                "input": matched_input,
                "most_similar_sentence": None,
                "distance": float(distance),
                "message": "No sufficiently similar sentence found."
            })

        return jsonify({
            "input": matched_input,
            "most_similar_sentence": most_similar,
            "distance": float(distance)
        })

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500

# 🔁 Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
