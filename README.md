# Fine-tuning a Sentence Transformer for Arxiv Research Paper Recommendations

This project demonstrates how to fine-tune a sentence-transformer model to recommend relevant research papers from the Arxiv dataset. The process involves two main steps:

1.  **Data Preparation:** Using the Gemini API to generate a labeled dataset of similar and dissimilar paper pairs.
2.  **Fine-tuning:** Using the generated dataset to fine-tune a pre-trained sentence-transformer model.

## Project Structure

-   `notebooks/`: Contains the Jupyter notebooks for data preparation and model fine-tuning.
    -   `01.Data_Preparation_(ASync)_Arxiv.ipynb`: This notebook uses the Gemini API to generate a labeled dataset of paper pairs with similarity scores.
    -   `02.Fine_tuning_sentence_transformer_03.ipynb`: This notebook fine-tunes a sentence-transformer model on the labeled dataset.
-   `README.md`: This file.

## Getting Started

### Prerequisites

-   Python 3.x
-   Jupyter Notebook or JupyterLab
-   A Google account with access to the Gemini API.
-   The Arxiv metadata dataset (`arxiv-metadata-oai-snapshot.json`).

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebooks

1.  **Data Preparation:**
    -   Open and run the `notebooks/01.Data_Preparation_(ASync)_Arxiv.ipynb` notebook.
    -   Make sure to set your `GEMINI_API_KEY` in the notebook.
    -   This notebook will generate a `my_labeled_training_data.csv` file.

2.  **Fine-tuning:**
    -   Open and run the `notebooks/02.Fine_tuning_sentence_transformer_03.ipynb` notebook.
    -   This notebook will load the `my_labeled_training_data.csv` file and fine-tune the sentence-transformer model.
    -   The fine-tuned model will be saved to a directory named `finetuned-arxiv-recommender`.

## How it Works

### Data Preparation

The `01.Data_Preparation_(ASync)_Arxiv.ipynb` notebook performs the following steps:

1.  **Loads Arxiv Metadata:** Reads the Arxiv metadata from a JSON file.
2.  **Generates Paper Pairs:** Creates random pairs of papers to be compared.
3.  **Gets Similarity Scores:** Uses the Gemini API to obtain a similarity score for each paper pair. The prompt instructs the model to act as a research assistant and provide a score between 0.0 and 1.0.
4.  **Saves Labeled Data:** Saves the paper pairs and their similarity scores to a CSV file.

### Fine-tuning

The `02.Fine_tuning_sentence_transformer_03.ipynb` notebook performs the following steps:

1.  **Loads Labeled Data:** Loads the `my_labeled_training_data.csv` file.
2.  **Prepares Data:** Converts the data into `InputExample` objects suitable for the `sentence-transformers` library.
3.  **Defines Model and Loss:** Uses the `all-MiniLM-L6-v2` model and `CosineSimilarityLoss`.
4.  **Fine-tunes:** Fine-tunes the model on the labeled dataset.
5.  **Saves Model:** Saves the fine-tuned model.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
