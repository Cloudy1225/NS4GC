# Reliable Node Similarity Matrix Guided Contrastive Graph Clustering

Official Implementation of Reliable Node Similarity Matrix Guided Contrastive Graph Clustering (TKDE 2024).



## Abstract

![Overview](./Overview.png)



## Dependencies

- PyTorch
- dgl (only as the source of datasets)
- scikit-learn



## Reproduction

Copy hyper-parameters from [params.txt](./params.txt) to [main.py](./main.py) and run it.



## Datasets

For all datasets, we use the processed version provided by [Deep Graph Library](https://github.com/dmlc/dgl).

| Dataset    | Type          | #Nodes | #Edges  | #Features | #Clusters | Homo   |
| ---------- | ------------- | ------ | ------- | --------- | --------- | ------ |
| Cora       | citation      | 2,708  | 10,556  | 1,433     | 7         | 82.52% |
| Citeseer   | citation      | 3,327  | 9,228   | 3,703     | 6         | 72.22% |
| Pubmed     | citation      | 19,717 | 88,651  | 500       | 3         | 79.24% |
| CoraFull   | citation      | 19,793 | 126,842 | 8,710     | 70        | 58.61% |
| WikiCS     | reference     | 11,701 | 431,726 | 300       | 10        | 65.88% |
| Photo      | co-purchase   | 7,650  | 238,163 | 745       | 8         | 83.65% |
| Computer   | co-purchase   | 13,752 | 491,722 | 767       | 10        | 78.53% |
| CoauthorCS | co-authorship | 18,333 | 163,788 | 6,805     | 15        | 83.20% |


