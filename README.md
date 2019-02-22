# DAE_RNN_News_Recommendation

Refer to research [Embedding-based News Recommendation for Millions of Users](https://www.kdd.org/kdd2017/papers/view/embedding-based-news-recommendation-for-millions-of-users)  and [Article De-duplication Using Distributed Representations](http://gdac.uqam.ca/WWW2016-Proceedings/companion/p87.pdf) published by Yahoo! Japan which aims to use embeddings of articles and user to provide news recommendations.

This repo contains code for training the article embeddings. And will include user embeddings in the future. 

Please check this [blog post](https://medium.com/@LouisKitLungLaw/compute-document-similarity-using-autoencoder-with-triplet-loss-eb7eb132eb38) for more details about the article embeddings training.

---

1. autoencoder.py contains code for normal denoising autoencoder (DAE) and DAE with triplet mining 
2. autoencoder_triplet.py contains code for DAE as describled in the paper
3. Data can be in the form of np.ndarray of scipy sparse matrix (e.g. csr_matrix)

---

Different training approaches implemented:
1. Normal denoising autoencoder (main_autoencoder.py with --triplet_strategy none) 
2. Triplet online mining. Positive and negative items are found automatically during training (main_autoencoder.py with --triplet_strategy batch_all or batch_hard)
3. The input data should include positive and negative items (main_autoencoder_triplet.py)

---
## Quick start

Python version = 3.6.7

```shell
# to train a new model
python main_autoencoder.py --model_name uci --verbose --encode_full

# to reuse previous model and previous training data and continue training
python main_autoencoder.py --model_name uci --verbose --encode_full --restore_previous_model --restore_previous_data
```

Monitor training in tensorboard

```shell
tensorboard --logdir results/dae/uci/logs
```

---
## Parameter

`main_autoencoder.py` could be run with following parameters

| Parameter  | Description | Default |
| ------------- | ------------- | ------------- |
| verbose  | Print log  | False |
| verbose_step  | Print log every x training step  | 5 |
| encode_full  | Whether to encode and store the embeddings after training | False |
| validation  | Whether to use a validation set and print validation loss during training  | False |
| input_format  | Two format available: binary vector or tfidf vector | binary |
| label  | Which labels is used to form triplet, category or story | category_publish_name |
| save_tsv  | Whether to save data in tsv format | False |
| train_row  | Number of training data | 8000 |
| validate_row  | Number of validation data | 2000 |
| restore_previous_data  | If true, restore previous data corresponding to model name  | False |
| min_df  | min_df for sklearn CountVectorizer | 0 |
| max_df  | max_df for sklearn CountVectorizer | 0.99 |
| max_features  | max_features for sklearn CountVectorizer | 10000 |
| model_name  | Model name |  |
| restore_previous_model  | If true, restore previous model corresponding to model name | False |
| seed  | Seed for the random generators (>= 0). Useful for testing hyperparameters | -1 |
| compress_factor  | Compression factor to determine num. of hidder nodes | 20 |
| corr_type  | Type of input corruption. ["none", "masking", "salt_and_pepper", "decay] | masking |
| corr_frac  | Fraction of the input to corrupt | 0.3 |
| xavier_init  | Value for the constant in xavier weights initialization | 1 |
| enc_act_func  | Activation function for the encoder. ["sigmoid", "tanh"] | sigmoid |
| dec_act_func  | Activation function for the decoder. ["sigmoid", "tanh", "none"] | sigmoid |
| main_dir  | Directory to store data relative to the algorithm. Same as model_name if empty | |
| loss_func  | Loss function of autoencoder ["mean_squared", "cross_entropy", "cosine_proximity"] | cross_entropy |
| opt  | Optimization algorithm ["gradient_descent", "ada_grad", "momentum"] | gradient_descent |
| learning_rate  | Initial learning rate | 0.1 |
| momentum  | Momentum parameter | 0.5 |
| num_epochs  | Number of epochs, set to 0 will not train the model | 50 |
| batch_size  | Size of each mini-batch, can be an integer or number between 0-1 | 0.1 |
| alpha  | hyper parameter for balancing similarity in loss function | 1 |
| triplet_strategy  | Triplet strategy used in triplet mining ["batch_all","batch_hard","none"] | batch_all |


## Acknowledgements

The autoencoder is modified based on git repo of [blackecho](https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9) and [Leavingseason](https://github.com/Leavingseason/rnn_recsys)  
Triplet online mining is referred from this [github](https://github.com/omoindrot/tensorflow-triplet-loss)
