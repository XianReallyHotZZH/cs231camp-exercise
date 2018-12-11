import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  - scores = X.dot(W)  N*C

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    # 计算每一个样本的得分 1*C
    score = np.dot(X[i], W)
    # 小技巧：防止后期求指数时候发生数值爆炸，将该样本所有的得分都减去最大值(数学上可以证明对最后的求loss是没有影响的)
    score -= max(score)
    # 分子：求指数（也可以叫做去对数）
    score = np.exp(score)
    # 分母：指数求和
    softmax_sum = np.sum(score)
    # 所有类别都求一下相应的概率
    score /= softmax_sum
    # 得到最终的损失（使用正确分类的概率进行计算）
    loss -= np.log(score[y[i]])

    # 梯度更新
    for j in range(num_class):
      if j!=y[i]:
        dW[:, j] += score[j]*X[i]
      else:
        dW[:, j] += (score[j] -1)*X[i]

  # 平均一下，防止数值爆炸
  loss /= num_train
  dW /= num_train
  # 加上正则损失
  loss += reg*np.sum(W*W)
  dW += 2 * reg * W

  
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)   # N*C

  # loss part
  # 小技巧：防止后期求指数时候发生数值爆炸，将该样本所有的得分都减去最大值(数学上可以证明对最后的求loss是没有影响的)
  #max_num = np.max(scores, axis=1)
  #scores_submax = (scores.T - np.max(scores, axis=1)).T
  scores_submax = scores - np.max(scores, axis=1).reshape(-1,1)
  # 分子：求指数（也可以叫做去对数）
  scores_fenzi = np.exp(scores_submax)
  # 分母：指数求和
  scores_fenmu = np.sum(scores_fenzi, axis=1)
  # 所有类别都求一下相应的概率(softmax值)
  #scores_softmax = (scores_fenzi.T / scores_fenmu).T
  scores_softmax = scores_fenzi / scores_fenmu.reshape(-1,1)
  # 得到最终的损失（使用正确分类的概率进行计算）
  loss -= np.sum( np.log(scores_softmax[range(num_train), y]) )

  # grad part(理解了naive里的dW更新后向量化即可)
  scores_softmax[range(num_train),y] -= 1
  dW = np.dot(X.T, scores_softmax)


  # 平均一下，防止数值爆炸
  loss /= num_train
  dW /= num_train
  # 加上正则损失
  loss += reg*np.sum(W*W)
  dW += 2 * reg * W



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

