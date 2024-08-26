# Translation Quality Evaluation

This project is designed to evaluate the quality of machine translations using BLEU and ROUGE scores. It compares translations from two different models (`Helsinki-NLP/opus-mt-tc-big-tr-en` and `Helsinki-NLP/opus-mt-tr-en`) and provides a detailed evaluation based on both performance and cost.

## Features
- **BLEU Score Calculation:** The BLEU score is used to measure the accuracy of the translation by comparing it to a reference translation.
- **ROUGE Score Calculation:** The ROUGE score measures the overlap of n-grams between the translation and reference, focusing on recall.
- **Cost Analysis:** The script estimates the cost of translation for each model based on the number of tokens.
- **Excel Output:** Results are saved in an Excel file with the best model highlighted for easy comparison.

## How It Works
1. The script fetches a reference translation for a given Turkish text using the MyMemory API.
2. It then generates translations using two different MarianMT models.
3. The quality of these translations is evaluated using BLEU and ROUGE scores.
4. The best translation is highlighted in the output Excel file, along with time taken and cost information.

## Installation
To run this project, you need to install the required Python libraries:

```bash
pip install gradio requests nltk rouge-score transformers pandas openpyxl

Usage
Run the translation_quality_evaluation.py script.
Enter the Turkish text in the input box of the Gradio interface.
The script will display the translation along with its cost.
The results will also be saved in an Excel file named translation_results.xlsx.
Example
Here is an example of how the Gradio interface looks:


Contribution
Feel free to fork this project, submit issues, or contribute through pull requests.