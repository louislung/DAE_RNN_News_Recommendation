"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(sparse_input, input_label, encode, pos_triplets_only = False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        input_label: labels of the batch, of size (batch_size,)
        encode: tensor of shape (batch_size, embed_dim)

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    # Get the dot product
    dotproduct = tf.matmul(encode, tf.transpose(encode))

    # shape (batch_size, batch_size, 1)
    anchor_positive_dotproduct = tf.expand_dims(dotproduct, 2)
    assert anchor_positive_dotproduct.shape[2] == 1
    # shape (batch_size, 1, batch_size)
    anchor_negative_dotproduct = tf.expand_dims(dotproduct, 1)
    assert anchor_negative_dotproduct.shape[1] == 1

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_distance = - anchor_positive_dotproduct + anchor_negative_dotproduct

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    valid_triplet_mask = tf.to_float(_get_triplet_mask(input_label))
    num_valid_triplets = tf.reduce_sum(valid_triplet_mask)

    # Count number of positive triplets (where triplet_distance > 0)
    pos_valid_triplet_mask = tf.to_float(tf.greater(tf.multiply(valid_triplet_mask, triplet_distance), 1e-16))
    num_pos_valid_triplets = tf.reduce_sum(pos_valid_triplet_mask)

    # Set final mask
    if pos_triplets_only:
        mask = pos_valid_triplet_mask
        num_triplet = num_pos_valid_triplets
    else:
        mask = valid_triplet_mask
        num_triplet = num_valid_triplets

    # Get final mean triplet loss over the (positive) valid triplets
    triplet_loss = - tf.log_sigmoid(-triplet_distance) * mask
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_triplet + 1e-16)

    data_weight = tf.reduce_sum(mask, [1, 2]) + tf.reduce_sum(mask, [0, 1]) + tf.reduce_sum(mask, [0, 2])

    return triplet_loss, data_weight, num_pos_valid_triplets / (num_valid_triplets + 1e-16), num_pos_valid_triplets


def batch_all_triplet_loss_org(sparse_input, input_label, encode, input_data, decode, pos_triplets_only = False, autoencoder_loss_func='cross_entropy'):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        input_label: labels of the batch, of size (batch_size,)
        encode: tensor of shape (batch_size, embed_dim)

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    # Get the dot product
    dotproduct = tf.matmul(encode, tf.transpose(encode))

    # shape (batch_size, batch_size, 1)
    anchor_positive_dotproduct = tf.expand_dims(dotproduct, 2)
    assert anchor_positive_dotproduct.shape[2] == 1
    # shape (batch_size, 1, batch_size)
    anchor_negative_dotproduct = tf.expand_dims(dotproduct, 1)
    assert anchor_negative_dotproduct.shape[1] == 1

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_distance = - anchor_positive_dotproduct + anchor_negative_dotproduct

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    valid_triplet_mask = tf.to_float(_get_triplet_mask(input_label))
    num_valid_triplets = tf.reduce_sum(valid_triplet_mask)

    # Count number of positive triplets (where triplet_distance > 0)
    pos_valid_triplet_mask = tf.to_float(tf.greater(tf.multiply(valid_triplet_mask, triplet_distance), 1e-16))
    num_pos_valid_triplets = tf.reduce_sum(pos_valid_triplet_mask)

    # Set final mask
    if pos_triplets_only:
        mask = pos_valid_triplet_mask
        num_triplet = num_pos_valid_triplets
    else:
        mask = valid_triplet_mask
        num_triplet = num_valid_triplets

    # Get final mean triplet loss over the (positive) valid triplets
    triplet_loss = - tf.log_sigmoid(-triplet_distance) * mask
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_triplet + 1e-16)

    data_weight = tf.reduce_sum(mask, [1, 2]) + tf.reduce_sum(mask, [0, 1]) + tf.reduce_sum(mask, [0, 2])

    # Autoencoder element wise cross entropy loss / mean squared loss
    _reduce_sum = tf.sparse.reduce_sum if sparse_input else tf.reduce_sum
    _to_dense = tf.sparse.to_dense if sparse_input else lambda x: x

    if autoencoder_loss_func == 'cross_entropy':
        autoencoder_loss = - tf.reduce_sum(_to_dense(input_data) * tf.log(decode+1e-16) + (1.-_to_dense(input_data)) * tf.log(1.-decode+1e-16), 1)
    elif autoencoder_loss_func == 'mean_squared':
        autoencoder_loss = tf.reduce_sum(tf.squared_difference(_to_dense(input_data),decode), 1)
    elif autoencoder_loss_func == 'cosine_proximity':
        autoencoder_loss = - tf.reduce_sum(tf.nn.l2_normalize(_to_dense(input_data),1) * tf.nn.l2_normalize(decode,1), 1)
    autoencoder_loss = tf.reduce_sum(autoencoder_loss * data_weight) / (tf.reduce_sum(data_weight) + 1e-16)
    # autoencoder_loss = tf.reduce_mean(autoencoder_loss)  # using this will make it becomes normal autoencoder loss

    return triplet_loss, autoencoder_loss, num_pos_valid_triplets / (num_valid_triplets + 1e-16), num_pos_valid_triplets


def batch_hard_triplet_loss(sparse_input, input_label, encode):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    # Get the pairwise distance matrix
    dotproduct = tf.matmul(encode, tf.transpose(encode))

    # For each anchor, get the hardest positive (similar items with smallest dotproduct)
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(input_label)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We add the maximum value in each row to the invalid positives (label(a) != label(n))
    max_anchor_dotproduct = tf.reduce_max(dotproduct, axis=1, keepdims=True)
    anchor_positive_dotproduct = dotproduct + max_anchor_dotproduct * (1.0 - mask_anchor_positive)

    # shape (batch_size, 1)
    hardest_positive_dotproduct = tf.reduce_min(anchor_positive_dotproduct, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dotproduct", tf.reduce_mean(hardest_positive_dotproduct))

    # For each anchor, get the hardest negative (dissimilar items with largest dotproduct)
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(input_label)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    anchor_negative_dotproduct = tf.multiply(mask_anchor_negative, dotproduct)

    # shape (batch_size,1)
    hardest_negative_dotproduct = tf.reduce_max(anchor_negative_dotproduct, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dotproduct", tf.reduce_mean(hardest_negative_dotproduct))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_dist = tf.maximum(hardest_negative_dotproduct - hardest_positive_dotproduct, 0.0)

    triplet_count = tf.to_float(tf.greater(triplet_dist,0.0))

    data_weight = tf.squeeze(triplet_count) + \
                  tf.reduce_sum(triplet_count * tf.to_float(tf.equal(dotproduct, hardest_positive_dotproduct)),0) + \
                  tf.reduce_sum(triplet_count * tf.to_float(tf.equal(dotproduct, hardest_negative_dotproduct)), 0)

    # Get final mean triplet loss
    triplet_loss = - tf.log_sigmoid(-triplet_dist) * triplet_count
    triplet_loss = tf.reduce_sum(triplet_loss) / (tf.reduce_sum(triplet_count) + 1e-16)

    return triplet_loss, data_weight, tf.reduce_sum(triplet_count) / tf.to_float(tf.shape(input_label)[0]), tf.reduce_sum(triplet_count)


def weighted_loss(sparse_input, input_data, decode, loss_func='cross_entropy', weight=None):
    _reduce_sum = tf.sparse.reduce_sum if sparse_input else tf.reduce_sum
    _to_dense = tf.sparse.to_dense if sparse_input else lambda x: x

    if weight is None: weight = tf.ones(tf.shape(input_data)[0])

    if loss_func == 'cross_entropy':
        autoencoder_loss = - tf.reduce_sum(_to_dense(input_data) * tf.log(decode+1e-16) + (1.-_to_dense(input_data)) * tf.log(1.-decode+1e-16), 1)
    elif loss_func == 'mean_squared':
        autoencoder_loss = tf.reduce_sum(tf.squared_difference(_to_dense(input_data),decode), 1)
    elif loss_func == 'cosine_proximity':
        autoencoder_loss = - tf.reduce_sum(tf.nn.l2_normalize(_to_dense(input_data),1) * tf.nn.l2_normalize(decode,1), 1)

    autoencoder_loss = tf.reduce_sum(autoencoder_loss * weight) / (tf.reduce_sum(weight) + 1e-16)

    return autoencoder_loss


if __name__ == '__main__':
    print('hi')
    # input_label = np.array([1,1,2]).astype('float32')
    # input_data = np.array([[1.1,1.2,1.3],[2.01,2.02,2.03],[3.01,3.02,3.03]]).astype('float32')
    # encode = np.array([[1.1,1.2,1.3],[2.01,2.02,2.03],[3.01,3.02,3.03]]).astype('float32')
    # decode = input_data
    # tf.Session().run(batch_hard_triplet_loss(False,input_label,input_data,encode,decode))

