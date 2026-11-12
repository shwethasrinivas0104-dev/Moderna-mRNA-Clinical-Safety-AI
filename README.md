Moderna mRNA Clinical Safety & Dosage Intelligence Platform (mRNA-VIP)
Project Overview
This project is an educational AI/ML platform designed to explore clinical safety analysis for mRNA vaccine products. It combines machine learning, deep learning, Retrieval-Augmented Generation (RAG), and an interactive dashboard.

Project Modules
Machine Learning: Predicts adverse-reaction risk using Logistic Regression and Random Forest.
Deep Learning: Implements a self-attention mechanism using NumPy to analyze sequence representations.
RAG Pipeline: Extracts, chunks, indexes, and retrieves information from a clinical safety reference guide using ChromaDB and sentence embeddings.
Clinical Report Generation: Generates structured JSON and Markdown safety reports using Pydantic validation.
Interactive Dashboard: Displays model evaluation, attention heatmaps, and RAG retrieval performance.
Technologies Used
Python
Pandas and NumPy
Scikit-learn
Matplotlib
Streamlit
ChromaDB
Sentence Transformers
Python-docx
Pydantic
Project Structure
data/ – Dataset and safety reference guide
src/ – Python source code
models/ – Trained machine learning model
outputs/ – Evaluation results and generated files
outputs/figures/ – Model and attention visualizations
outputs/reports/ – Generated clinical safety reports
How to Run
Clone this repository.
Install dependencies using pip install -r requirements.txt.
Run the dashboard using:
streamlit run src/dashboard.py
Disclaimer
This is an educational project using sample data and a fictional safety reference guide. It is not clinically validated and must not be used for medical diagnosis, treatment, or real-world dosing decisions.

Author
AI/ML Healthcare project

AI/ML Healthcare Project

