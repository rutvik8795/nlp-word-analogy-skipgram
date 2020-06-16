import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    batch_size = inputs.get_shape().as_list()
    batch_size = batch_size[0]
    mulTensor = tf.multiply(inputs, true_w)
    sumTensor = tf.reduce_sum(mulTensor, axis=1)
    sumTensor = tf.where(tf.is_nan(sumTensor), tf.zeros_like(sumTensor), sumTensor)
    sumTensor = tf.reshape(sumTensor, [batch_size, 1])

    A = tf.log(tf.exp(sumTensor))

    mulTensorB = tf.matmul(inputs, true_w, transpose_b=True)
    expTensorB = tf.exp(mulTensorB)
    reducedSumB = tf.reduce_sum(expTensorB, axis=1)
    B = tf.reshape(tf.log(reducedSumB), [batch_size, 1])

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    sample_size = len(sample)
    input_size = inputs.get_shape().as_list()
    batch_size = input_size[0]
    embedding_size = input_size[1]

    targetEmbeddings = tf.nn.embedding_lookup(weights, labels, name="TargetWordEmbedding")
    targetEmbeddings = tf.reshape(targetEmbeddings, [batch_size, embedding_size])  # [batch_size, embedding_size]
    unigramProb = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    unigramProbabilities = tf.nn.embedding_lookup(unigramProb, labels, name="TargetUnigramProbs")  # [batch_size, 1]

    targetBias = tf.nn.embedding_lookup(biases, labels, name="TargetBias")  # [batch_size, 1]

    dotProduct = tf.multiply(inputs, targetEmbeddings)
    reducedSum = tf.reduce_sum(dotProduct , axis=1)
    score = tf.reshape(reducedSum, [batch_size, 1])  # [batch_size, 1]

    score = tf.add(score, targetBias)

    samples = tf.convert_to_tensor(sample, dtype=tf.int32)

    scalarMul = tf.scalar_mul(sample_size, unigramProbabilities)
    smallValToPreventNan = [0.0000000001 for j in range(batch_size)]
    smallValToPreventNanTensor = tf.reshape(tf.convert_to_tensor(smallValToPreventNan, dtype=tf.float32), [batch_size, 1])
    addSmallNum = tf.add(scalarMul, smallValToPreventNanTensor)
    unigramProbabilities = tf.log(addSmallNum)  # [batch_size,1]

    score = tf.subtract(score, unigramProbabilities)  # [batch_size, 1]
    score = tf.sigmoid(score)
    score = tf.log(score)  # [batch_size, 1]

    negTargetEmbeddings = tf.nn.embedding_lookup(weights, samples,name="NegativeTargetEmbedding")  # [sample_size, embedding_size]

    negEmbeddings = tf.nn.embedding_lookup(unigramProb, sample, name="NegativeUnigramProbs")
    reshapedNegEmbeddings = tf.reshape(negEmbeddings, [sample_size, 1])
    negUniProb = tf.transpose(reshapedNegEmbeddings)

    negTargetEmbeddingBias = tf.nn.embedding_lookup(biases, sample, name="NegativeTargetBias")
    negTargetEmbeddingBias = tf.reshape(negTargetEmbeddingBias, [sample_size,1])  # [sample_size, 1]
    negTargetEmbeddingBias = tf.transpose(negTargetEmbeddingBias)  # [1, sample_size]
    negTargetEmbeddingBias = tf.tile(negTargetEmbeddingBias, [batch_size, 1])  # [batch_size, sample_size]

    negScore = tf.matmul(inputs, negTargetEmbeddings, transpose_b=True)  # [batch_size, sample_size]

    negScore = tf.add(negScore, negTargetEmbeddingBias)  # [batch_size, sample_size]

    scalarMulNegProb = tf.scalar_mul(sample_size, negUniProb)
    logNegProb = tf.log(scalarMulNegProb)  # [1, sample_size]
    negUniProb = tf.tile(logNegProb,[batch_size, 1])  # [batch_size, sample_size]

    negScore = tf.subtract(negScore, negUniProb)
    unitMat = [[1.0 for j in range(sample_size)] for i in range(batch_size)]
    negScore = tf.subtract(unitMat, tf.sigmoid(negScore))

    smallValToPreventNan2 = [[0.0000000001 for j in range(sample_size)] for i in range(batch_size)]
    smallValToPreventNanTensor2 = tf.reshape(tf.convert_to_tensor(smallValToPreventNan2, dtype=tf.float32), [batch_size, sample_size])
    addSmallNumNeg = tf.add(negScore, smallValToPreventNanTensor2)
    logNeg = tf.log(addSmallNumNeg)
    reducedSumNeg = tf.reduce_sum(logNeg, axis=1)
    negScore = tf.reshape(reducedSumNeg, [batch_size, 1])

    sampleProb = tf.scalar_mul(-1, tf.add(score, negScore))

    return sampleProb
