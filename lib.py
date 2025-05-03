import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import os
import subprocess
import sys

# For Seq2Seq and fine-tuned models
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import evaluate
    transformers_available = True
except ImportError:
    transformers_available = False
    print("Transformers library not available. Install with: pip install transformers evaluate torch")

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextSummarizer:
    def __init__(self, glove_path='glove.6B.100d.txt', device=None):
        """
        Initialize the summarizer with all three approaches
        
        Args:
            glove_path: Path to GloVe embeddings
            device: Device to use for PyTorch models ('cuda', 'cpu', or None for auto-detect)
        """
        self.stop_words = stopwords.words('english')
        self.word_embeddings = {}
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu') if transformers_available else 'cpu'
        
        # Load GloVe embeddings if file exists
        if os.path.exists(glove_path):
            print(f"Loading word embeddings from {glove_path}...")
            with open(glove_path, encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.word_embeddings[word] = coefs
            print(f"Loaded {len(self.word_embeddings)} word vectors.")
        else:
            print(f"GloVe embeddings not found at {glove_path}. Extractive summarization will be limited.")
        
        # Initialize Seq2Seq and fine-tuned models if transformers is available
        if transformers_available:
            try:
                print("Setting up T5 model for Seq2Seq summarization...")
                self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
                self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
                
                print("Setting up BART model for fine-tuned summarization...")
                self.bart_summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    device=0 if self.device == 'cuda' else -1
                )
                print("Models loaded successfully.")
            except Exception as e:
                print(f"Error loading transformer models: {e}")
        
    def download_glove_embeddings(self, output_dir='.'):
        """Download GloVe embeddings if not available."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        glove_path = os.path.join(output_dir, 'glove.6B.100d.txt')
        zip_path = os.path.join(output_dir, 'glove.6B.zip')
        
        if not os.path.exists(glove_path):
            if not os.path.exists(zip_path):
                print("Downloading GloVe embeddings...")
                subprocess.run(
                    ["wget", "http://nlp.stanford.edu/data/glove.6B.zip", "-P", output_dir],
                    check=True
                )
            
            print("Extracting GloVe embeddings...")
            subprocess.run(
                ["unzip", "-o", zip_path, "glove.6B.100d.txt", "-d", output_dir],
                check=True
            )
            
        return glove_path
    
    def remove_stopwords(self, sentence):
        """Remove stopwords from a sentence."""
        return " ".join([word for word in sentence if word not in self.stop_words])
    
    def preprocess_text(self, text):
        """Preprocess text for extractive summarization."""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Clean sentences
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
        clean_sentences = [s.lower() for s in clean_sentences]
        
        # Remove stopwords
        clean_sentences = [self.remove_stopwords(s.split()) for s in clean_sentences]
        
        return sentences, clean_sentences
    
    def get_sentence_vectors(self, clean_sentences):
        """Convert sentences to vectors using word embeddings."""
        sentence_vectors = []
        for sentence in clean_sentences:
            if len(sentence) != 0:
                vector = sum([self.word_embeddings.get(w, np.zeros((100,))) for w in sentence.split()]) / (len(sentence.split()) + 0.001)
            else:
                vector = np.zeros((100,))
            sentence_vectors.append(vector)
        return sentence_vectors
    
    def extractive_summarize(self, text, num_sentences=3):
        """
        Generate extractive summary using TextRank algorithm.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to include in summary
            
        Returns:
            str: Extractive summary
        """
        # Preprocess text
        sentences, clean_sentences = self.preprocess_text(text)
        
        # Get sentence vectors
        sentence_vectors = self.get_sentence_vectors(clean_sentences)
        
        # Create similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(
                        sentence_vectors[i].reshape(1, 100),
                        sentence_vectors[j].reshape(1, 100)
                    )[0, 0]
        
        # Apply PageRank
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        
        # Get top sentences
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)),
            reverse=True
        )
        
        # Return joined summary
        return " ".join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    
    def seq2seq_summarize(self, text, max_length=150):
        """
        Generate summary using Seq2Seq (T5) model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of generated summary
            
        Returns:
            str: Generated summary
        """
        if not transformers_available:
            return "Transformers library not available. Install with: pip install transformers torch"
        
        try:
            # Prepare input
            inputs = self.t5_tokenizer("summarize: " + text, 
                                      max_length=512, 
                                      truncation=True, 
                                      return_tensors="pt").to(self.device)
            
            # Generate summary
            summary_ids = self.t5_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            return self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating Seq2Seq summary: {e}"
    
    def fine_tuned_summarize(self, text, max_length=150, min_length=30):
        """
        Generate summary using fine-tuned BART model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            
        Returns:
            str: Generated summary
        """
        if not transformers_available:
            return "Transformers library not available. Install with: pip install transformers torch"
        
        try:
            return self.bart_summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
        except Exception as e:
            return f"Error generating fine-tuned summary: {e}"
    
    def evaluate_summaries(self, reference_texts, generated_summaries):
        """
        Evaluate summaries using ROUGE metrics.
        
        Args:
            reference_texts: List of reference summaries
            generated_summaries: List of generated summaries
            
        Returns:
            dict: ROUGE scores
        """
        if not transformers_available:
            return "Evaluation requires the 'evaluate' library. Install with: pip install evaluate"
        
        try:
            rouge = evaluate.load('rouge')
            scores = rouge.compute(
                predictions=generated_summaries,
                references=reference_texts,
                rouge_types=['rouge1', 'rouge2', 'rougeL']
            )
            return scores
        except Exception as e:
            return f"Error evaluating summaries: {e}"


def process_dataset(summarizer, dataset_path):
    """Process a dataset of articles and generate summaries."""
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path, encoding='latin1')
        print(f"Dataset loaded successfully with {len(df)} articles.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using sample article instead.")
        # Sample text if dataset fails to load
        sample_text = """
        Maria Sharapova has basically no friends as tennis players on the WTA Tour. The Russian player has no problems in openly speaking about it and in a recent interview she said: 'I don't really hide any feelings too much. I think everyone knows this is my job here. When I'm on the courts or when I'm on the court playing, I'm a competitor and I want to beat every single person whether they're in the locker room or across the net. So I'm not the one to strike up a conversation about the weather and know that in the next few minutes I have to go and try to win a tennis match. I'm a pretty competitive girl. I say my hellos, but I'm not sending any players flowers as well. Uhm, I'm not really friendly or close to many players. I have not a lot of friends away from the courts.'
        """
        # Create a dummy dataframe
        df = pd.DataFrame({
            'article_id': [1],
            'article_text': [sample_text]
        })
    
    # Process a subset of articles (use 5 for demonstration)
    num_articles = min(5, len(df))
    articles = df.iloc[:num_articles]
    
    # Lists to store all results
    article_ids = []
    extractive_summaries = []
    seq2seq_summaries = []
    bart_summaries = []
    
    # Process each article
    for idx, row in articles.iterrows():
        article_id = row.get('article_id', idx)
        article_text = row['article_text']
        
        print(f"\n\n{'='*80}")
        print(f"Article #{article_id}")
        print(f"{'='*80}")
        
        # Print the first 200 characters of the article
        print(f"Original Text (truncated):\n{article_text[:200]}...\n")
        
        # Generate and store extractive summary
        print("Generating extractive summary...")
        extractive_summary = summarizer.extractive_summarize(article_text, 3)
        extractive_summaries.append(extractive_summary)
        print(f"Extractive Summary:\n{extractive_summary}\n")
        
        if transformers_available:
            # Generate and store seq2seq summary
            print("Generating Seq2Seq summary...")
            seq2seq_summary = summarizer.seq2seq_summarize(article_text)
            seq2seq_summaries.append(seq2seq_summary)
            print(f"Seq2Seq Summary:\n{seq2seq_summary}\n")
            
            # Generate and store BART summary
            print("Generating fine-tuned BART summary...")
            bart_summary = summarizer.fine_tuned_summarize(article_text)
            bart_summaries.append(bart_summary)
            print(f"Fine-tuned Summary:\n{bart_summary}\n")
        else:
            seq2seq_summaries.append("N/A - Transformers not available")
            bart_summaries.append("N/A - Transformers not available")
        
        # Store article ID
        article_ids.append(article_id)
    
    # Evaluate summaries if transformers are available
    if transformers_available and len(extractive_summaries) > 0:
        print("\n\n" + "="*80)
        print("Evaluation Comparison")
        print("="*80)
        
        # For evaluation purposes, we'll use the extractive summaries as reference
        # In a real-world scenario, you would use human-written summaries as reference
        reference_summaries = extractive_summaries
        
        # Evaluate each method
        print("\nEvaluating with ROUGE metrics...")
        
        # Compare extractive vs. seq2seq
        print("\nSeq2Seq vs. Extractive:")
        seq2seq_scores = summarizer.evaluate_summaries(
            reference_summaries, 
            seq2seq_summaries
        )
        print(seq2seq_scores)
        
        # Compare extractive vs. BART
        print("\nBART vs. Extractive:")
        bart_scores = summarizer.evaluate_summaries(
            reference_summaries, 
            bart_summaries
        )
        print(bart_scores)
        
        # Compare seq2seq vs. BART
        print("\nBART vs. Seq2Seq:")
        bart_vs_seq2seq = summarizer.evaluate_summaries(
            seq2seq_summaries, 
            bart_summaries
        )
        print(bart_vs_seq2seq)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'article_id': article_ids,
        'extractive_summary': extractive_summaries,
        'seq2seq_summary': seq2seq_summaries,
        'bart_summary': bart_summaries
    })
    
    results_path = 'summarization_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}") 