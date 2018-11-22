# DAE_RNN_News_Recommendation

Refer to research [Embedding-based News Recommendation for Millions of Users](https://www.kdd.org/kdd2017/papers/view/embedding-based-news-recommendation-for-millions-of-users) published by Yahoo! Japan which aims to use embeddings of articles and user to provide news recommendations.

---

1. autoencoder.py contains code for normal denoising autoencoder (DAE)
2. autoencoder_triplet.py contains code for DAE as describled in the paper
3. Data can be in the form of np.ndarray of scipy sparse matrix (e.g. csr_matrix)

---
Quick start

Python version = 3.6.7

```shell
# to train a new model
python main.py --model_name dae_triplet --verbose --encode_full

# to reuse previous model and previous training data and continue training
python main.py --model_name dae_triplet --verbose --encode_full --restore_previous_model --restore_previous_data
```

---
The autoencoder is modified based on git repo of [blackecho](https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9) and [Leavingseason](https://github.com/Leavingseason/rnn_recsys)
