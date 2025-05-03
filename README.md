# Text Summarizer

A comprehensive text summarization tool that implements three different approaches to summarize text:

1. **Extractive Summarization** (TextRank/PageRank-based)
2. **Sequence-to-Sequence Summarization** (T5-small)
3. **Fine-tuned Summarization** (BART-large-CNN)

## Project Structure

```
text_summarizer/
├── lib.py                 # Core TextSummarizer class and processing functions
├── app.py                 # Flask web application
├── templates/             # HTML templates
│   └── index.html         # Main web interface
└── README.md              # This file
```

## Requirements

- Python 3.6+
- Flask
- NLTK
- NumPy
- pandas
- scikit-learn
- networkx
- Transformers (for Seq2Seq and fine-tuned models)
- PyTorch (for Seq2Seq and fine-tuned models)

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install flask nltk numpy pandas scikit-learn networkx transformers torch evaluate
```

3. Download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Download GloVe embeddings (will be done automatically on first run)

## Usage

### Web Interface

To start the web interface:

```bash
python app.py
```

Then open a browser and navigate to http://localhost:5000

### Process a CSV Dataset

To process a dataset of articles:

```bash
python app.py --dataset path/to/your/dataset.csv
```

The dataset should have an 'article_text' column containing the text to summarize.

### Command Line Options

```
--dataset PATH    Path to dataset CSV file
--port PORT       Port for web interface (default: 5000)
--host HOST       Host for web interface (default: 0.0.0.0)
--debug           Run Flask in debug mode
```

## TextSummarizer API

You can also use the TextSummarizer class directly in your Python code:

```python
from lib import TextSummarizer

# Initialize the summarizer
summarizer = TextSummarizer()

# Generate summaries
text = "Your text to summarize..."
extractive_summary = summarizer.extractive_summarize(text, num_sentences=3)
seq2seq_summary = summarizer.seq2seq_summarize(text, max_length=150)
bart_summary = summarizer.fine_tuned_summarize(text, max_length=150, min_length=30)

# Print summaries
print("Extractive Summary:", extractive_summary)
print("Seq2Seq Summary:", seq2seq_summary)
print("BART Summary:", bart_summary)
```

## Summary Approach Comparison

1. **Extractive Summarization (TextRank/PageRank)**
   - **Strengths**: Preserves original text, fast processing, no training required
   - **Limitations**: Only selects existing sentences, may miss context between sentences

2. **Seq2Seq Summarization (T5-small)**
   - **Strengths**: Can generate new sentences, better handles overall context
   - **Limitations**: May generate inaccurate information, less fluent than BART

3. **Fine-tuned Summarization (BART-large-CNN)**
   - **Strengths**: Most fluent and coherent summaries, best at capturing key points
   - **Limitations**: Most computationally expensive, may introduce hallucinations

## License

MIT 