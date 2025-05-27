# 🔬 streamlit-pubmed

A **Streamlit app** for searching PubMed, retrieving relevant biomedical literature, and exploring how machine learning models (e.g., for classification or similarity) can be trained using scientific text.

This tool provides an intuitive interface to explore scientific literature and visually prototype machine learning workflows. A separate **Jupyter Notebook** is included for users who want to implement and customize full model training pipelines.

---

## 🚀 Features

* 🔎 **Search PubMed**: Enter any biomedical topic or query and retrieve top relevant articles via the PubMed API.
* 📑 **Article Viewer**: Display titles, abstracts, and metadata in a clean, readable format.
* 📊 **Modeling Workflow Overview**: In-app guidance and visualizations showing how the retrieved data could be used to train models.
* 📓 **Notebook for Training**: Use the provided Jupyter Notebook to train an actual **XGBoost** model on the fetched articles (e.g., for classification or similarity tasks).

---

## ▶️ Run the App

Use the following command to start the Streamlit app:

```bash
streamlit run medical_app.py
```

This will open a new tab in your default browser. If it doesn't, visit the printed `localhost` URL manually.

---

## 💡 Jupyter Notebook: Model Training

The `train_model.ipynb` notebook contains an implementation of an **XGBoost** training pipeline using article metadata and content. This notebook is intended for users who want to:

* Go beyond the Streamlit UI
* Customize feature engineering, preprocessing, and evaluation

To run it:

```bash
jupyter notebook train_model.ipynb
```

---

## 🧪 Sample Use Case

1. Search for: `"cancer immunotherapy"`
2. View top abstracts from PubMed in the Streamlit app.
3. Export or use selected articles in the `model_training.ipynb` notebook.
4. Train an XGBoost classifier using article labels or categories.

---

## 📚 Tech Stack

* [Streamlit](https://streamlit.io/)
* [PubMed Entrez API](https://www.ncbi.nlm.nih.gov/home/develop/api/)
* [Sentence Transformers](https://www.sbert.net/)
* [XGBoost](https://xgboost.readthedocs.io/)
* [scikit-learn](https://scikit-learn.org/)
* [Jupyter Notebook](https://jupyter.org/)
