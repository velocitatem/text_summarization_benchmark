from flask import Flask, request, render_template, jsonify
import argparse
import os
import sys

# Add the current directory to path to import lib.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib import TextSummarizer, process_dataset, transformers_available

app = Flask(__name__)

# Initialize the summarizer
summarizer = TextSummarizer()

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint for summarizing text."""
    data = request.json
    text = data.get('text', '')
    num_sentences = int(data.get('num_sentences', 3))
    max_length = int(data.get('max_length', 150))
    
    # Generate summaries
    extractive_summary = summarizer.extractive_summarize(text, num_sentences)
    
    if transformers_available:
        seq2seq_summary = summarizer.seq2seq_summarize(text, max_length)
        bart_summary = summarizer.fine_tuned_summarize(text, max_length)
    else:
        seq2seq_summary = "Transformers library not available. Install with: pip install transformers torch"
        bart_summary = "Transformers library not available. Install with: pip install transformers torch"
    
    return jsonify({
        'extractive_summary': extractive_summary,
        'seq2seq_summary': seq2seq_summary,
        'bart_summary': bart_summary
    })

@app.route('/about')
def about():
    """About page with information about the summarization techniques."""
    return render_template('about.html')

def main():
    """Main function to either process dataset or launch web interface."""
    parser = argparse.ArgumentParser(description="Text Summarization Tool")
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV file')
    parser.add_argument('--port', type=int, default=5000, help='Port for web interface')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for web interface')
    parser.add_argument('--debug', action='store_true', help='Run Flask in debug mode')
    
    args = parser.parse_args()
    
    # Try to download GloVe if not available
    if not os.path.exists('glove.6B.100d.txt'):
        try:
            summarizer.download_glove_embeddings()
        except Exception as e:
            print(f"Error downloading GloVe embeddings: {e}")
            print("Please download them manually from http://nlp.stanford.edu/data/glove.6B.zip")
    
    if args.dataset:
        # Process dataset instead of starting the web server
        process_dataset(summarizer, args.dataset)
    else:
        # Start web interface
        print(f"Web interface running at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 