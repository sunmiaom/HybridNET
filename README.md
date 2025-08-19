# HybridNET
 Provided Files:
HybridNET.py：data_process：Preprocessing module for data preparation before similarity calculations，Adjust_cosine-similar：Analyzes parent and product spectra data to compute adjusted cosine similarity，Cosine-similarityAnalyzes parent and product spectra data to compute cosine similarity ，Dot-similarity：Analyzes parent and product spectra data to compute dot product similarity
Entropy-similarity：Analyzes parent and product spectra data to compute entropy similarity，Neutral loss-similarity：Analyzes parent and product spectra data to compute neutral loss similarity，similar_evaluate：Evaluates and compares the performance of all five similarity algorithms

1. Requirements
The program requires the following Python libraries:
- numpy
- pandas  
- ms_entropy
- networkx
- matplotlib
- tqdm

Installation command:
```bash
pip install numpy pandas ms_entropy networkx matplotlib tqdm
```

---

2. Important Notes

Input Data Requirements:
- The script requires two input files in the working directory:
  - `parent.xlsx` - Parent spectra data
  - `product.xlsx` - Product spectra data

Parameter Configuration:
- Similarity threshold can be adjusted by modifying the condition:
  ```python
  if similarity > 0.5  # Default threshold set to 0.5 for identifying similar spectra
  ```

---

 3. Output Files

Primary Output:
- `similarities.csv` - Contains computed similarity scores for each parent-product spectra pair
