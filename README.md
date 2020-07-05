# Avocado Classifier | Logistic regression with a neural network model

## Purpose statement

The purpose of this project is: 
1. to theoretically examine the extent to which a logistic regression model with a neural network can classify images. 
2. to gain practical experience with the following: 
   - preprocessing and converting image data into vector arrays using `PIL`; 
   - creating, reshaping, broadcasting and performing mathematical operations with vector arrays using `NumPy`; and
   - combining the different parts of a shallow neural network algorithm into a class.

## Description of data

The data comes from a directory called [fruit-360 | Kaggle](https://www.kaggle.com/moltean/fruits) and consists of the following features: 
- 297 images were used in total, with 167 images of avocados, and 130 images of other miscellaneous fruits and vegetables. 

![Image 7-4-20 at 6 37 AM](https://user-images.githubusercontent.com/43279348/86510783-e4e7fa00-bdc0-11ea-92d2-ae778780c22f.jpg)

- each square image is of shape (100, 100, 3) where 3 is for the 3 channels on the RGB scale.  

![Image 7-4-20 at 6 45 AM](https://user-images.githubusercontent.com/43279348/86510908-25944300-bdc2-11ea-90cb-70bc1108c024.jpg)

## Description of algorithm  

The building blocks for logistic regression with a neural network algorithm consist of the following:

**I.  Initialize weight and bias parameters, *w* and *b*.**

**II.  Instantiate forwad propagation to compute cost.** 

   For each training example, *i*, compute: 

<a href="https://www.codecogs.com/eqnedit.php?latex=(1)&space;\quad&space;\quad&space;z^{(i)}&space;=w^{T}x^{(i)}&plus;b,&space;\mbox{where}\&space;x^{(i)}&space;\&space;\mbox{is&space;the&space;$x$&space;feature&space;vector&space;for&space;training&space;example&space;$i$}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?(1)&space;\quad&space;\quad&space;z^{(i)}&space;=w^{T}x^{(i)}&plus;b,&space;\mbox{where}\&space;x^{(i)}&space;\&space;\mbox{is&space;the&space;$x$&space;feature&space;vector&space;for&space;training&space;example&space;$i$}." title="(1) \quad \quad z^{(i)} =w^{T}x^{(i)}+b, \mbox{where}\ x^{(i)} \ \mbox{is the $x$ feature vector for training example $i$}." /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=(2)&space;\quad&space;\quad&space;\hat{y}^{(i)}&space;=&space;a^{(i)}&space;=&space;\sigma&space;(z^{(i)})&space;=&space;\frac{1}{1&space;-&space;e^{-(z^{(i)})}},\&space;\mbox{where}\&space;\hat{y}^{(i)}\&space;\mbox{is&space;the&space;predicted&space;y-output&space;value&space;for&space;training&space;example&space;$i$.}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(2)&space;\quad&space;\quad&space;\hat{y}^{(i)}&space;=&space;a^{(i)}&space;=&space;\sigma&space;(z^{(i)})&space;=&space;\frac{1}{1&space;-&space;e^{-(z^{(i)})}},\&space;\mbox{where}\&space;\hat{y}^{(i)}\&space;\mbox{is&space;the&space;predicted&space;y-output&space;value&space;for&space;training&space;example&space;$i$.}" title="(2) \quad \quad \hat{y}^{(i)} = a^{(i)} = \sigma (z^{(i)}) = \frac{1}{1 - e^{-(z^{(i)})}},\ \mbox{where}\ \hat{y}^{(i)}\ \mbox{is the predicted y-output value for training example $i$.}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=(3)&space;\quad&space;\quad&space;{\cal{L}}&space;(a^{(i)},&space;y^{(i)})&space;=&space;-y^{(i)}\log&space;(a^{(i)})&space;-&space;(1-y^{(i)})\log&space;(1-a^{(i)})&space;\mbox{,&space;where&space;$&space;y^{(i)}$&space;is&space;the&space;vector&space;containg&space;$y$&space;labels&space;for&space;each&space;training&space;example&space;$i$.}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(3)&space;\quad&space;\quad&space;{\cal{L}}&space;(a^{(i)},&space;y^{(i)})&space;=&space;-y^{(i)}\log&space;(a^{(i)})&space;-&space;(1-y^{(i)})\log&space;(1-a^{(i)})&space;\mbox{,&space;where&space;$&space;y^{(i)}$&space;is&space;the&space;vector&space;containg&space;$y$&space;labels&space;for&space;each&space;training&space;example&space;$i$.}" title="(3) \quad \quad {\cal{L}} (a^{(i)}, y^{(i)}) = -y^{(i)}\log (a^{(i)}) - (1-y^{(i)})\log (1-a^{(i)}) \mbox{, where $ y^{(i)}$ is the vector containg $y$ labels for each training example $i$.}" /></a>

Then, compute the cost function by taking the average of *(3)* for all training examples: 

<a href="https://www.codecogs.com/eqnedit.php?latex=(4)&space;\quad&space;\quad&space;J=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log&space;(a^{(i)})&space;-&space;(1-y^{(i)})\log&space;(1-a^{(i)})\mbox{,&space;where&space;$m$&space;is&space;the&space;total&space;number&space;of&space;training&space;examples.}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(4)&space;\quad&space;\quad&space;J=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log&space;(a^{(i)})&space;-&space;(1-y^{(i)})\log&space;(1-a^{(i)})\mbox{,&space;where&space;$m$&space;is&space;the&space;total&space;number&space;of&space;training&space;examples.}" title="(4) \quad \quad J=-\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log (a^{(i)}) - (1-y^{(i)})\log (1-a^{(i)})\mbox{, where $m$ is the total number of training examples.}" /></a>

**III.  Instantiate backward propagation to compute gradient descent.** 

Compute the derivatives of *w* and *b*:

<a href="https://www.codecogs.com/eqnedit.php?latex=(5)&space;\quad&space;\quad&space;\frac{dJ}{dw}=\frac{1}{m}X(A-Y)^{T},\&space;\mbox{where&space;$X$,&space;$A$,&space;and&space;$Y$&space;are&space;matrices&space;containing&space;$x$,&space;$a$,&space;and&space;$y$&space;vectors&space;for&space;the&space;entire&space;training&space;set.}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(5)&space;\quad&space;\quad&space;\frac{dJ}{dw}=\frac{1}{m}X(A-Y)^{T},\&space;\mbox{where&space;$X$,&space;$A$,&space;and&space;$Y$&space;are&space;matrices&space;containing&space;$x$,&space;$a$,&space;and&space;$y$&space;vectors&space;for&space;the&space;entire&space;training&space;set.}" title="(5) \quad \quad \frac{dJ}{dw}=\frac{1}{m}X(A-Y)^{T},\ \mbox{where $X$, $A$, and $Y$ are matrices containing $x$, $a$, and $y$ vectors for the entire training set.}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=(6)&space;\quad&space;\quad&space;\frac{dJ}{db}=\frac{1}{m}\sum_{i=1}^{m}(a^{(i)}-y^{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(6)&space;\quad&space;\quad&space;\frac{dJ}{db}=\frac{1}{m}\sum_{i=1}^{m}(a^{(i)}-y^{(i)})" title="(6) \quad \quad \frac{dJ}{db}=\frac{1}{m}\sum_{i=1}^{m}(a^{(i)}-y^{(i)})" /></a>

**IV.  Update weight and bias parameters, using gradient descent to minimize the cost function.** 

Update *w* and *b* parameters by computing *(7)* and *(8)*, where α is the learning rate:

<a href="https://www.codecogs.com/eqnedit.php?latex=(7)&space;\quad&space;\quad&space;w&space;=&space;w&space;-&space;(\alpha&space;*dw)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(7)&space;\quad&space;\quad&space;w&space;=&space;w&space;-&space;(\alpha&space;*dw)" title="(7) \quad \quad w = w - (\alpha *dw)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=(8)&space;\quad&space;\quad&space;b&space;=&space;b&space;-&space;(\alpha&space;*db)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(8)&space;\quad&space;\quad&space;b&space;=&space;b&space;-&space;(\alpha&space;*db)" title="(8) \quad \quad b = b - (\alpha *db)" /></a>

**V.  Iterate over II-IV to optimize your parameters.**





