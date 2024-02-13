# On the Privacy of Sublinear-Communication Jaccard Index Estimation via Min-hash Sketching

This repository is the official implementation of [On the Privacy of Sublinear-Communication Jaccard Index Estimation via Min-hash Sketching].

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The experiment is implemented based on Python3, and should have pip version>=22.0.2. 

## To run the experiment, run: 

```experiment
python3 main.py
```

Graph will be generated and stored for each section. 
To adjust the size and scale or the outcome graph, e.g. 8x6 graph size, use "fig, ax = plt.subplots(figsize=(8, 6))" at each graph function. 
The experiment takes a while to finish, especially the one for "number of iterations vs jaccard index" for public hash setting (*MinhashGraphPBinomJaccardVsK()*), 
due to the usage of binary search to find the optimal number of iterations. 
Hence, our suggestion is to run MinhashGraphPBinom() instead, which provides "epsilon vs delta" giving intersection size and Jaccard index, 
to get a quick assessment of the trade-off between intersection size, Jaccard index, and privacy parameters.
Experiment raw results, including the case of n=100k and n=1million, is also included as an Excel chart with graphs. 

## Results

Below are the empirical evaluation results of parameters in both curator setting or public hash setting. See the section 6 in our paper for detail. 

The results for the curator setting: 

eps, delta: 
![Figure](/Pre_run_figures/Binom_eps_vs_delta.png)

Iteration vs Jaccard: 
![Figure](/Pre_run_figures/B_JK_1mil.png)


The results for the public hash setting: 

eps, delta: 
![Figure](/Pre_run_figures/PBinom_eps_vs_delta.png)

Iteration vs Jaccard: 
![Figure](/Pre_run_figures/PB_JK_1mil.png)