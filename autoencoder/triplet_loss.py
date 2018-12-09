"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf, numpy as np


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


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


def batch_all_triplet_loss(sparse_input, input_label, input_data, encode, decode):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

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

    # shape (batch_size, batch_size, 1)
    anchor_positive_dotproduct = tf.expand_dims(dotproduct, 2)
    assert anchor_positive_dotproduct.shape[2] == 1, "{}".format(anchor_positive_dotproduct.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dotproduct = tf.expand_dims(dotproduct, 1)
    assert anchor_negative_dotproduct.shape[1] == 1, "{}".format(anchor_negative_dotproduct.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_negative_dotproduct - anchor_positive_dotproduct

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(input_label)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    positive_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(positive_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = - tf.log_sigmoid(-triplet_loss) * positive_triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    # Autoencoder element wise cross entropy loss
    _reduce_sum = tf.sparse.reduce_sum if sparse_input else tf.reduce_sum

    cross_entropy_count = tf.reduce_sum(positive_triplets, [1, 2]) + tf.reduce_sum(positive_triplets, [0, 1]) + tf.reduce_sum(positive_triplets, [0, 2])
    cross_entropy_loss = -_reduce_sum(input_data * tf.log(tf.clip_by_value(decode,1e-16,1.)), 1)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss * cross_entropy_count)/(tf.reduce_sum(cross_entropy_count) + 1e-16)
    return cross_entropy_loss, triplet_loss, fraction_positive_triplets, num_positive_triplets


def batch_hard_triplet_loss(sparse_input, input_label, input_data, encode, decode):
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

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(input_label)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We add the maximum value in each row to the invalid positives (label(a) != label(n))
    max_anchor_dotproduct = tf.reduce_max(dotproduct, axis=1, keepdims=True)
    anchor_positive_dotproduct = dotproduct + max_anchor_dotproduct * (1.0 - mask_anchor_positive)

    # shape (batch_size, 1)
    hardest_positive_dotproduct = tf.reduce_min(anchor_positive_dotproduct, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dotproduct", tf.reduce_mean(hardest_positive_dotproduct))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(input_label)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    anchor_negative_dotproduct = tf.multiply(mask_anchor_negative, dotproduct)

    # shape (batch_size,)
    hardest_negative_dotproduct = tf.reduce_max(anchor_negative_dotproduct, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dotproduct", tf.reduce_mean(hardest_negative_dotproduct))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_dist = tf.maximum(hardest_negative_dotproduct - hardest_positive_dotproduct, 0.0)

    # Autoencoder element wise cross entropy loss
    _reduce_sum = tf.sparse.reduce_sum if sparse_input else tf.reduce_sum

    triplet_count = tf.to_float(tf.greater(triplet_dist,0.0))
    cross_entropy_count = tf.squeeze(triplet_count) + \
                          tf.reduce_sum(triplet_count * tf.to_float(tf.equal(dotproduct,hardest_positive_dotproduct)),0) + \
                          tf.reduce_sum(triplet_count * tf.to_float(tf.equal(dotproduct,hardest_negative_dotproduct)),0)
    cross_entropy_loss = -_reduce_sum(input_data * tf.log(tf.clip_by_value(decode, 1e-16, 1.)), 1)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss * cross_entropy_count) / (tf.reduce_sum(cross_entropy_count) + 1e-16)

    # Get final mean triplet loss
    triplet_loss = - tf.log_sigmoid(-triplet_dist) * triplet_count
    triplet_loss = tf.reduce_sum(triplet_loss) / (tf.reduce_sum(triplet_count) + 1e-16)

    return cross_entropy_loss, triplet_loss, tf.reduce_sum(triplet_count) / tf.to_float(tf.shape(input_label)[0]), tf.reduce_sum(triplet_count)


if __name__ == '__main__':
    def test_simple_batch_all_triplet_loss():
        """Test the triplet loss with batch all triplet mining in a simple case.

        There is just one class in this super simple edge case, and we want to make sure that
        the loss is 0.
        """
        num_data = 5
        input_dim = 20
        feat_dim = 6
        num_classes = 1

        input_data = np.random.rand(num_data, input_dim).astype(np.float32)
        encode = np.random.rand(num_data, feat_dim).astype(np.float32)
        print(encode.mean(1))
        input_label = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)
        #input_label = np.array([_ for _ in range(0,num_classes)])

        corss_entropy_loss_np = 0.0
        triplet_loss_np = 0.0

        # Compute the loss in TF.
        cross_entropy_loss, triplet_loss = batch_all_triplet_loss(False, input_label, input_data, encode, input_data)
        with tf.Session() as sess:
            cross_entropy_loss_val, triplet_loss_val = sess.run([cross_entropy_loss, triplet_loss])
        print(triplet_loss_val)
        assert np.allclose(corss_entropy_loss_np, cross_entropy_loss_val)
        assert np.allclose(triplet_loss_np, triplet_loss_val)


    test_simple_batch_all_triplet_loss()


    input_label = np.array([1,1,2]).astype('float32')
    input_data = np.array([[1.1,1.2,1.3],[2.01,2.02,2.03],[3.01,3.02,3.03]]).astype('float32')
    encode = np.array([[1.1,1.2,1.3],[2.01,2.02,2.03],[3.01,3.02,3.03]]).astype('float32')
    decode = input_data
    tf.Session().run(batch_hard_triplet_loss(False,input_label,input_data,encode,decode))

