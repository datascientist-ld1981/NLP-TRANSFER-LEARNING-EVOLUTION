# ML to LLM Sentiment Analysis

This repository contains the source code and dataset for exploring sentiment analysis using traditional machine learning models and transitioning to advanced techniques leveraging large language models (LLMs). The project highlights the evolution of NLP techniques and provides comparative insights.

## Objectives

1. **Study Evolution**: Trace the development of sentiment analysis models, from Naive Bayes to transfer learning models.
2. **Comparative Analysis**: Analyze model architectures in depth, comparing their performance and limitations.
3. **Descriptive vs Generative Models**: Explore BERT and GPT for sentiment analysis on the IMDB dataset, focusing on time efficiency, resource usage, and contextual complexities.

## Repository Structure

```
├── ml_to_llm.ipynb   # Main Jupyter Notebook
├── dataset/          # Dataset files (e.g., IMDB reviews)
├── README.md         # Project documentation
└── results/          # Output and evaluation results
```

## Key Features

- Implementation of Naive Bayes, RNN, LSTM, BERT, and GPT for sentiment analysis.
- Comparative analysis of traditional and transformer-based models.
- Benchmarking on the IMDB dataset.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow`/`pytorch`, `transformers`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/datascientist-ld1981/ml-to-llm.git
   cd ml-to-llm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook

1. Open `ml_to_llm.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Follow the step-by-step implementation and modify parameters as needed.

## Results

The project provides:

- Performance metrics for each model (accuracy, F1-score, etc.).
- Resource usage comparisons (training time, memory, etc.).
- Insights into contextual understanding by descriptive and generative models.

## Future Work

- Extend analysis to other datasets.
- Experiment with ensemble models combining traditional ML and LLM approaches.
- Explore domain-specific fine-tuning for sentiment analysis.

## Author

**Lakshmi Devi**  
[![GitHub](https://img.shields.io/badge/GitHub-datascientist--ld1981-blue?logo=github)](https://github.com/datascientist-ld1981)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face Transformers for BERT and GPT implementations.
- Scikit-learn for ML model support.
- TensorFlow and PyTorch for deep learning frameworks.
- IMDB dataset for benchmark testing.

