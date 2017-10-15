import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    incr = 0
    
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        incr += 1
        dW[:, j] += X[i].T # Incorrect class
        loss += margin
  
    dW[:, y[i]] += -incr * X[i].T # Correct class
   
    """
    # This works, was to try out
    for j in range(num_classes):
      xi = X[i] # 1 x 3073
      if j == y[i]:
        # Gradient for correct class
        scale = 0
        
        for j1 in range(num_classes):
          if j1 == y[i]:
            continue
          margin = scores[j1] - correct_class_score + 1 # note delta = 1
          if margin > 0:
            scale += 1
            
        dW[:, j] += (-1) * scale * xi.T
      else:
        # Gradient for rest
        scale = 0
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
            scale = 1
        dW[:, j] += scale * xi.T
    """
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW = dW/num_train + reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  # Done, see above

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  
  scores_all = X.dot(W)
  scores_correct = scores_all[np.arange(len(scores_all)), y].reshape((1, -1)).T
  scores = scores_all - scores_correct + 1
  scores[scores < 0] = 0
  scores[np.arange(len(scores)), y] = 0
  # Now the scores matrix is num_train x num_classes
  # For each row, the correct class will have value of 0. Any -ve score also 
  # would have turned to 0.
  # Update correct class:
  
    
  loss = np.sum(scores) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  scores[scores>0] = 1
  scores[np.arange(num_train),y] = -np.sum(scores,axis=1)
  dW = X.T.dot(scores)

  dW = dW/num_train 
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
