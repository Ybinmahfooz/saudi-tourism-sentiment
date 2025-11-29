# Saudi Tourism Sentiment Analysis (Arabic)

This project analyzes sentiment (Positive, Neutral, Negative) for Arabic tourism reviews using:

- **Classical ML models**: Logistic Regression, SVM, Decision Tree, Random Forest  
- **Deep learning models**: AraBERT, mBERT, XLM-R  
- **Word2Vec embeddings**  
- **CLI demo** to test your own Arabic sentences

## Project Structure

- `preprocessing_and_eda.py` – load dataset, clean text, visualizations (bar chart + word cloud)  
- `classical_models.py` – Word2Vec encoding + classical ML models  
- `deep_models.py` – AraBERT, mBERT, XLM-R training + AraBERT fine-tuning  
- `main.py` – full pipeline (calls preprocessing + classical + deep models)  
- `demo.py` – interactive demo to test sentences  
- `data/` – dataset files (ignored in git)  
- `models/` – trained models (ignored in git)  
- `logs/`, `results/` – training logs, plots and metrics  

## Setup

```bash
# create and activate vertual environment
python -m venv venv
venv\Scripts\activate 

# install dependencies
pip install -r requirements.txt

# train models + generate plots and logs
python main.py

# run interactive demo
python demo.py
