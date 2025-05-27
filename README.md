# 🔬 streamlit-pubmed

A **Streamlit app** for searching PubMed, retrieving relevant biomedical literature, and training a **custom similarity model** based on the retrieved articles.

This tool provides an intuitive interface to explore scientific literature and build machine learning models for text similarity, all without writing a single line of code.

---

## 🚀 Features

* 🔎 **Search PubMed**: Enter any biomedical topic or query and retrieve top relevant articles via the PubMed API.
* 📑 **Article Viewer**: Display titles, abstracts, and metadata in a clean, readable format.
---

## ▶️ Run the App

Use the following command to start the Streamlit app:

```bash
streamlit run medical_app.py
```

This will open a new tab in your default browser. If not, go to the printed `localhost` URL.

---

## 🧪 Sample Use Case

1. Search for: `"cancer immunotherapy"`
2. View top abstracts from PubMed.
3. Click to select useful articles.

---

## 📚 Tech Stack

* [Streamlit](https://streamlit.io/)
* [PubMed Entrez API](https://www.ncbi.nlm.nih.gov/home/develop/api/)
* [Sentence Transformers](https://www.sbert.net/)
* [scikit-learn](https://scikit-learn.org/)
