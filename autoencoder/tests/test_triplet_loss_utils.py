from ..triplet_loss_utils import _get_triplet_mask
from ..triplet_loss_utils import _get_anchor_positive_triplet_mask
from ..triplet_loss_utils import _get_anchor_negative_triplet_mask
from ..triplet_loss_utils import batch_all_triplet_loss
from ..triplet_loss_utils import batch_hard_triplet_loss
from ..triplet_loss_utils import weighted_loss
import numpy as np, tensorflow as tf, pytest
from sklearn.preprocessing import normalize

# todo: add testing for sparse_input = True
@pytest.mark.parametrize("classes", [1, 3, 5])
def test_get_all_triplet_mask(classes):
    num_data = 5
    num_classes = classes

    input_label = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    batch_all_mask_np = np.zeros((num_data,num_data,num_data)).astype(bool)

    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                if i == j or j == k or i == k: continue
                if input_label[i] == input_label[j] and input_label[i] != input_label[k]:
                    batch_all_mask_np[i, j, k] = True

    batch_all_mask = tf.Session().run(_get_triplet_mask(input_label))

    assert (batch_all_mask == batch_all_mask_np).all()
    print('Function [_get_triplet_mask] passed for classes={}'.format(classes))

@pytest.mark.parametrize("classes", [1, 3, 5])
def test_get_anchor_positive_triplet_mask(classes):
    num_data = 5
    num_classes = classes

    input_label = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data)).astype(bool)

    for i in range(num_data):
        for j in range(num_data):
            if i == j: continue
            if input_label[i] == input_label[j]:
                mask_np[i, j] = True

    mask = tf.Session().run(_get_anchor_positive_triplet_mask(input_label))

    assert (mask == mask_np).all()
    print('Function [_get_anchor_positive_triplet_mask] passed for classes={}'.format(classes))

@pytest.mark.parametrize("classes", [1, 3, 5])
def test_get_anchor_negative_triplet_mask(classes):
    num_data = 5
    num_classes = classes

    input_label = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data)).astype(bool)

    for i in range(num_data):
        for j in range(num_data):
            if i == j: continue
            if input_label[i] != input_label[j]:
                mask_np[i, j] = True

    mask = tf.Session().run(_get_anchor_negative_triplet_mask(input_label))

    assert (mask == mask_np).all()
    print('Function [test_get_anchor_negative_triplet_mask] passed for classes={}'.format(classes))

@pytest.mark.parametrize("classes", [1, 3, 5])
def test_batch_all_triplet_loss(classes):
    """Test the triplet loss with batch all triplet mining in a simple case.

    Suggest to test classed >= 2 and classes = 1 (for extreme case where all items are in same cate)
    """
    num_data = 20
    input_dim = 20
    feat_dim = 6
    num_classes = classes

    #
    # Test cross entropy loss, input data is one hot binary count vectorized
    #
    input_data = np.random.randint(0,2,(num_data, input_dim)).astype(np.float32)
    encode = np.random.rand(num_data, feat_dim).astype(np.float32)
    decode = np.random.rand(num_data, input_dim).astype(np.float32)
    input_label = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)
    dotproduct = encode.dot(encode.transpose())

    valid_triplet_mask = tf.Session().run(_get_triplet_mask(input_label))
    pos_valid_triplet_mask = valid_triplet_mask.copy()

    all_valid_triplet_loss_np = np.zeros((num_data,num_data,num_data))
    valid_data_weight_np = np.zeros(num_data)
    valid_triplet_loss_np = 0.
    pos_valid_data_weight_np = np.zeros(num_data)
    pos_valid_triplet_loss_np = 0.

    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                if i == j or j == k or i == k: continue
                if input_label[i] == input_label[j] and input_label[i] != input_label[k]:
                    # valid triplet
                    distance = dotproduct[i, k] - dotproduct[i, j]
                    loss = np.log1p(np.exp(distance))
                    all_valid_triplet_loss_np[i, j, k] = loss

                    valid_data_weight_np[[i,j,k]] += 1
                    valid_triplet_loss_np += loss

                    if distance > 1e-16:
                        # valid and positive triplet
                        pos_valid_data_weight_np[[i, j, k]] += 1
                        pos_valid_triplet_loss_np += loss
                    else:
                        pos_valid_triplet_mask[i, j, k] = False

    valid_triplet_loss_np = valid_triplet_loss_np / (valid_triplet_mask.sum() + 1e-16)
    pos_valid_triplet_loss_np = pos_valid_triplet_loss_np / (pos_valid_triplet_mask.sum() + 1e-16)

    cross_entropy_loss_np = -(input_data*np.log(decode+1e-16) + (1-input_data)*np.log(1-decode+1e-16)).sum(1)

    # Compute the loss in TF.
    triplet_loss_val, data_weight, fraction_pos_valid_triplets, num_pos_valid_triplets = tf.Session().run(batch_all_triplet_loss(False, input_label, encode, False))
    assert (np.allclose(valid_triplet_loss_np, triplet_loss_val))
    assert (np.allclose(valid_data_weight_np, data_weight))
    assert (np.allclose(pos_valid_triplet_mask.sum()/(valid_triplet_mask.sum() + 1e-16), fraction_pos_valid_triplets))
    assert (np.allclose(pos_valid_triplet_mask.sum(), num_pos_valid_triplets))

    # Compute the loss in TF.
    triplet_loss_val, data_weight, fraction_pos_valid_triplets, num_pos_valid_triplets = tf.Session().run(batch_all_triplet_loss(False, input_label, encode, True))
    assert (np.allclose(pos_valid_triplet_loss_np, triplet_loss_val))
    assert (np.allclose(pos_valid_data_weight_np, data_weight))

    print('Function [batch_all_triplet_loss] passed for classes={}'.format(classes))

@pytest.mark.parametrize("classes", [1, 3, 5])
def test_batch_hard_triplet_loss(classes):
    """Test the triplet loss with batch all triplet mining in a simple case.

    Suggest to test classed >= 2 and classes = 1 (for extreme case where all items are in same cate)
    """
    num_data = 20
    input_dim = 20
    feat_dim = 6
    num_classes = classes

    #
    # Test input data is one hot binary count vectorized
    #
    input_data = np.random.randint(0,2,(num_data, input_dim)).astype(np.float32)
    encode = np.random.rand(num_data, feat_dim).astype(np.float32)
    decode = np.random.rand(num_data, input_dim).astype(np.float32)
    input_label = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)
    dotproduct = encode.dot(encode.transpose())

    # valid_triplet_mask = tf.Session().run(_get_triplet_mask(input_label))
    # pos_valid_triplet_mask = valid_triplet_mask.copy()

    hardest_pos = np.array([np.nan] *num_data)
    hardest_pos_idx = np.array([np.nan] *num_data)
    hardest_neg = np.array([np.nan] *num_data)
    hardest_neg_idx = np.array([np.nan] *num_data)
    data_weight_np = np.zeros(num_data)
    triplet_loss_np = 0.
    num_triplet_np = 0

    for i in range(num_data):
        for j in range(num_data):
            if i == j: continue
            if input_label[i] == input_label[j]:
                if np.isnan(hardest_pos[i]) or dotproduct[i,j] < hardest_pos[i]:
                    hardest_pos[i] = dotproduct[i,j]
                    hardest_pos_idx[i] = j
            elif input_label[i] != input_label[j]:
                if np.isnan(hardest_neg[i]) or dotproduct[i,j] > hardest_neg[i]:
                    hardest_neg[i] = dotproduct[i,j]
                    hardest_neg_idx[i] = j

    triplet_hardest_distance = hardest_neg - hardest_pos
    for idx, val in enumerate(triplet_hardest_distance):
        if val > 0:
            data_weight_np[idx] += 1
            data_weight_np[int(hardest_pos_idx[idx])] += 1
            data_weight_np[int(hardest_neg_idx[idx])] += 1
            triplet_loss_np += np.log1p(np.exp(val))
            num_triplet_np += 1
        else:
            triplet_hardest_distance[idx] = np.nan

    triplet_loss_np = triplet_loss_np /  (num_triplet_np + 1e-16)

    # Compute the loss in TF.
    triplet_loss_val, data_weight, fraction_triplets, num_triplets = tf.Session().run(batch_hard_triplet_loss(False, input_label, encode))
    assert (np.allclose(triplet_loss_np, triplet_loss_val)),(triplet_loss_np, triplet_loss_val)
    assert (np.allclose(data_weight_np, data_weight))
    assert (np.allclose(num_triplet_np/num_data, fraction_triplets))
    assert (np.allclose(num_triplet_np, num_triplets))

    print('Function [batch_hard_triplet_loss] passed for classes={}'.format(classes))

def test_weighted_loss():
    num_data = 20
    input_dim = 20

    input_data = np.random.randint(0, 2, (num_data, input_dim)).astype(np.float32)
    decode = np.random.rand(num_data, input_dim).astype(np.float32)
    data_weight = np.random.randint(0,50,num_data).astype(np.float32)

    # Test cross_entropy
    cross_entropy_loss_np = -(input_data * np.log(decode + 1e-16) + (1. - input_data) * np.log(1. - decode + 1e-16)).sum(1)
    cross_entropy_loss = tf.Session().run(weighted_loss(False,input_data, decode, loss_func='cross_entropy'))
    assert (np.allclose(cross_entropy_loss_np.mean(), cross_entropy_loss))
    cross_entropy_loss = tf.Session().run(weighted_loss(False, input_data, decode, loss_func='cross_entropy', weight=data_weight))
    assert (np.allclose((cross_entropy_loss_np*data_weight).sum()/(data_weight.sum()), cross_entropy_loss))

    # Test mean_squared
    mean_squared_loss_np = np.square(input_data - decode).sum(1)
    mean_squared_loss = tf.Session().run(weighted_loss(False, input_data, decode, loss_func='mean_squared'))
    assert (np.allclose(mean_squared_loss_np.mean(),mean_squared_loss))
    mean_squared_loss = tf.Session().run(weighted_loss(False, input_data, decode, loss_func='mean_squared', weight=data_weight))
    assert (np.allclose((mean_squared_loss_np * data_weight).sum() / (data_weight.sum()), mean_squared_loss))

    # Test cosine_proximity
    cosine_loss_np = -(normalize(input_data, axis=1) * normalize(decode, axis=1)).sum(1)
    cosine_loss = tf.Session().run(weighted_loss(False, input_data, decode, loss_func='cosine_proximity'))
    assert (np.allclose(cosine_loss_np.mean(),cosine_loss))
    cosine_loss = tf.Session().run(weighted_loss(False, input_data, decode, loss_func='cosine_proximity', weight=data_weight))
    assert (np.allclose((cosine_loss_np * data_weight).sum() / (data_weight.sum()), cosine_loss))

    print('Function [test_weighted_loss] passed')
