Rutvik Parekh
SBU ID: 112687483


NLP Assignment 1 
Word2Vec: skipgrams with cross-entropy loss and NCE
README
Implementation Details


1. Batch Generation:
First, the word-ids are are extracted from the ‘data’ array by accessing the data_index.
Then, two indices left_index and right_index are kept to keep track of the left and right side words of the current window. 
When, the num_skips is reached in the current window, the data_index is incremented by 1. In this way, we keep shifting the window until we have reached the batch_size.


The batch variable is a list which keeps track of the context words and the label variable is a list which keeps track of the target words. For each batch element, the label variable contains the corresponding target word.


Here, one main point to focus is that the num_skips is lesser than or equal to 2 times the value of skip_windows, which means that not all the target words are considered in the given window. Once, the num_skips get over, we move to the immediate right of the context word.


The left and right words in a given window are selected starting from the center as the context word. 


The batches generated and the corresponding labels are provided to the training model for generating the learning model.


























2. Cross Entropy Loss: 
Cross entropy is a loss function used to measure the distance between two probability distributions. It returns zero if two distributions match exactly, and higher values if the two diverge. This is the negative of the log likelihood.


First, batch size is extracted from the input vector. 


Calculation of A
Then, we take the dot product of the context embeddings and target embeddings.
An important operation to be performed here is to remove the NaN values. 
I use tf.where() function to do that. After that, the dot product tensor is reshaped to the size of [batch_size,1] to return as in the output format.
Then, to calculate the value of A tensor, we take the log of the dot product.        


Calculation of B
        It’s most similar to most of the parts in A. But, here we multiply the matrices of input and         the target words ( true_w). This is done to consider all the combinations of the context
        And the target words. We take the exp of it by tf.exp() and then reduce it to [batch_size,1] 
        By using the method tf.reduce_sum(). The log of that value is taken.


        B-A is returned as the final value.


        The best accuracy ( 33.9 %) is achieved on the word_analogy_dev.txt file by setting the following 
        Hyper-parameter configuration:
        Batch_Size: 64
        Num_skips: 8
        Skip_windows: 4
        Max_num_steps: 300000






        The model which achieved the best result can be accessed at the following Google Drive
        Link: 
        https://drive.google.com/file/d/1VuEnMgcQ9y4OoIUKumRHvASVrEFoUjro/view?usp=sharing








3. NCE 
Noise contrastive estimation is implemented by subtracting the probabilities of all the noisy words from the probability of the true target words. Parameters such as inputs which are the embeddings, weights, biases, labels, samples and unigram probabilities 
are provided to the method. First, we extract the target embeddings, the target unigram probabilities and the target biases from the parameters to a tensor. Those are extracted by a function tf.nn.embedding_lookup. We calculate the first part of the equation by taking the dot product of context and target word embeddings. We then calculate the log likelihood of the unigram probabilities and then subtract it from the dot product and then take the sigmoid of the whole thing and then take the log of that value.


For the negative part, we repeat the process. But, in the end, we reduce the matrix to a sum and then subtract it from 1. Then we reshape it to [batch_size, 1] and add it to the previous part and multiply it by -1.


Here, one important point to note is to add some small value (0.0000000001) to prevent NaN values.








The best accuracy ( 35.9 %) is achieved on the word_analogy_dev.txt file by setting the following  hyper-parameter configuration:


        Batch_Size: 128
        Num_skips: 16
        Skip_windows: 8
        Max_num_steps: 300000
        Learning Rate: 25.0


The model which achieved the best result can be accessed at the following Google Drive
        Link: 


https://drive.google.com/file/d/1vD7XpyZR7Xz4tX8hnolqULDdQxS9ZCmr/view?usp=sharing
























4. Word Analogy Task
        In this task, we read the dev and the test file. It contains a few pair of words. 
        The format of the dev file is: 3 word-pairs || 4 option word-pairs


        The two words in the 3 pairs on the left side has some kind of relation between them.
        The relation is almost similar among all the pairs. We need to find the pair on the right
        Side of the two pipes which has the most similar relation and the least similar relation as 
        Illustrated by the pairs on the left side of the two pipes.


        First, the model is loaded and then the important information such as dictionary and 
        embeddings generated from the model. Then we separate the data provided in the file 
        Python list structures by removing all redundant information from the file.


        Two structures word_pairs and option_pairs are created.
1. Word_pairs contains the pair of words on the left side of the two pipes 
2. Option_pairs contains the pair of words on the right side of the two pipes.


        Then, the difference between the embeddings of each pair in word_pair is calculated to 
        Reflect the relation between the words in each pair. Those difference in the embeddings
        Are stored in diffWords. We then calculate the mean of the differences and store it in 
        diffWordMean. Similarly, we calculate the differences between the options pairs and 
        Store the difference in optionsDiff. 
        We compute the similarity between each of the options and the diffWordsMean by
             Cosine similarity method. We calculate the cosine similarity by using the function from 
        The spatial package .The function is spatial.distance.cosine(). We subtract the cosine 
        Distance from 1 to get the similarity. Then, we iterate among the four pairs to find the
        Most and the least illustrative pairs by taking the maximum cosine similarity for the 
        Former and the minimum cosine similarity for the latter. We then dump the output to the 
        Prediction file. In the format of 
        <Option_pair1> <option_pair2> <option_pair3> <option_pair4> <most illustrative pair> 
        <least illustrative pair>