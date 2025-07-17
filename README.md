# HybridNET
Provided files:  
data_process.py: Used for data processing before similarity calculation.  
Adjust_cosine-similar.py: Used to analyze spectral data of parent and product compounds, calculate adjusted cosine similarity, and visualize the generated molecular network.  
Cosine-similarity.py: Used to analyze spectral data of parent and product compounds, calculate cosine similarity, and visualize the generated molecular network.  
Dot-similarity.py: Used to analyze spectral data of parent and product compounds, calculate dot product similarity, and visualize the generated molecular network.  
Entropy-similarity.py: Used to analyze spectral data of parent and product compounds, calculate entropy similarity, and visualize the generated molecular network.  
Neutral_loss-similarity.py: Used to analyze spectral data of parent and product compounds, calculate neutral loss similarity, and visualize the generated molecular network.  
similar_evaluate.py: Used to evaluate and analyze the performance of the five similarity algorithms.  

 1. Requirements:  
This program requires the following Python libraries: 'numpy', 'pandas', 'ms_entropy', 'networkx', 'matplotlib', and 'tqdm'.  
You can install them using the following command:  
```ruby  
pip install numpy pandas ms_entropy networkx matplotlib tqdm  
```  

 2. Notes:  
Input Data: The script requires the presence of 'parent.xlsx' and 'product.xlsx' files in the working directory. When running 'similar_evaluate.py', the input data consists of the outputs from the five similarity calculations.  
Parameters: You can adjust the similarity threshold by modifying the line 'if similarity > 0.5' in the code. The default threshold for identifying similar spectra is set to 0.5. Adjust the network layout and graph plotting parameters according to your needs.  

 3. Output:  
similarities.csv: This file contains the calculated similarity scores for each pair of parent and product spectra.  
network.png: A molecular network graph plotted based on the calculated similarities.
