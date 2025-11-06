# Mastering the AI Toolkit: A Multi-Model Showcase

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mnisthandwrittendigitclassifier.streamlit.app/)

This repository contains the complete submission for the "Mastering the AI Toolkit" group assignment. The project demonstrates proficiency in key AI/ML frameworks, including **Scikit-learn**, **TensorFlow**, and **spaCy**.

The main deliverable, a handwritten digit classifier, is built with a TensorFlow CNN and deployed as an interactive web app using Streamlit.

## 🚀 Live Demo

**Test the live application here:**
### [https://mnisthandwrittendigitclassifier.streamlit.app/](https://mnisthandwrittendigitclassifier.streamlit.app/)

---

## 📚 Assignment Components

This repository contains all parts of the assignment:

* **Part 1: Theoretical Understanding**
    * All theoretical questions and comparative analyses are answered in the `report.pdf`.

* **Part 2: Practical Implementation**
    * **Task 1 (Classical ML):** `task1_iris_classifier.ipynb`
        * A Decision Tree classifier built with **Scikit-learn** to predict Iris species.
    * **Task 2 (Deep Learning):** `task2_mnist_cnn.ipynb`
        * A Convolutional Neural Network (CNN) built with **TensorFlow/Keras** to classify MNIST handwritten digits with >98% accuracy.
        * This notebook also generates the `mnist_cnn_model.h5` file used by the Streamlit app.
    * **Task 3 (NLP):** `task3_spacy_ner.ipynb`
        * A **spaCy** pipeline to perform Named Entity Recognition (NER) and rule-based sentiment analysis on sample Amazon reviews.

* **Part 3: Ethics & Optimization**
    * **Ethical Considerations:** A detailed analysis of potential bias in the NLP model is included in the `report.pdf`.
    * **Troubleshooting Challenge:** `part3_troubleshooting.py` contains the debugged and corrected TensorFlow code.

* **Bonus Task: Streamlit Deployment**
    * `bonus_app.py`: The Python script that powers the live, interactive web application.
    * `mnist_cnn_model.h5`: The saved, trained TensorFlow model.

---

## 🛠️ Tech Stack

* **Core Frameworks:** TensorFlow, Keras, Scikit-learn, spaCy
* **Data & Ops:** Pandas, NumPy, OpenCV (for image processing)
* **Deployment:** Streamlit
* **Development:** Jupyter Notebooks, Google Colab

---

## 💻 How to Run This Project Locally

To run the web app on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [Your-GitHub-Repo-URL]
    cd [your-repo-name]
    ```

2.  **Install all required libraries:**
    (It's recommended to use a virtual environment)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the spaCy language model:**
    (This is needed for Task 3)
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run bonus_app.py
    ```
    Your browser will automatically open to the app.

---

## Theory Answers

* **Final Report:** `[report.pdf]`

---
