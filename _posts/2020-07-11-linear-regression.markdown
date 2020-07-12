---
layout: post
title:  "Linear Regression"
date:   2020-07-11 22:50:16 -0400
categories: jekyll update
---

# Linear regression
First the problem should be posed mathematically. The dataset is defined as $\mathcal{D} = (X_i,Y_i)_{i=1}^n$ where  $X_i$ represents all the input features of some datapoints and $Y_i$ the coresponding features to predict.  $Y_i$ is asumed to be gaussian with some noise which means that $Y_i = w^{\intercal}X_i + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0,\sigma^2)$



### Offset notation
The offset notation is used for $x$. That means that $x = \begin{pmatrix} \tilde{x} \\ 1\end{pmatrix}$ where $\tilde{x} \in \mathbf{R}^{d-1}$ and the 1 is for the constant features. This means that $\langle w,x\rangle = \langle w_{1:d-1},\tilde{x}\rangle + w_d$ where $w_d$ is the "bias/offset". 

# Conditional Likelihood
$$p(y_{1:n} \mid x_{1:n}) = \prod\limits_{i=1}^n p(y_{i} \mid x_{i})$$

It is hard to optimize a multiplication, so we instead optimize the $\log$ of that function that is defined by $l(w) = \log p(y_{1:n} \mid x_{1:n})$. Use the log transform the multiplicaiton into a sum which is much easier to manipulate. It is not a problem to use the log function because it is a monotone function, meaning that it is always increasing so a a maximum in the original function will also ba a maximum in the log of that function.

$$
\begin{align*}
    \log p(y_{1:n}|x_{1:n})  & = \log \prod\limits_{i=1}^n p(y_{i}|x_{i}) \\
    & = \sum\limits_{i=1}^n \log p(y_{i}|x_{i} )
\end{align*}
$$

Now we find a simplified expression for $l(w)$

$$
\begin{align*}
    l(w)  & =  \sum\limits_{i=1}^n \log p(y_{i}|x_{i} )\\
    & = \sum\limits_{i=1}^n \log \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(y_{i} -w^{\intercal}x_{i})^2}{2\sigma ^2}}\\
    & = \sum\limits_{i=1}^n -\frac{(y_{i} -w^{\intercal}x_{i})^2}{2\sigma^2}-\frac{1}{2} \log(2\pi\sigma^2) \\
    & = \frac{-1}{2} \sum\limits_{i=1}^n \left[ \frac{(y_{i} -w^{\intercal}x_{i})^2}{\sigma^2}+ \log(2\pi\sigma^2) \right]
\end{align*}
$$


# Design matrix
The Design Matrix $X$ enables the use of linear algebra to solve for $w$

$$ X \triangleq \left[ \begin{array}{c} x^{\intercal}_{1} \\ \vdots \\ x^{\intercal}_{n} \end{array} \right] \in \mathbb{R}^{n \times d} $$,
$$ y \triangleq \left[\begin{array}{c}y_{1}\\\vdots \\y_{n} \end{array}\right] \in \mathbb{R}^{n \times 1}$$,
$$
Xw = \left[ \begin{array}{c} x^{\intercal}_{1}w  \\ \vdots    \\  x^{\intercal}_{n}w  \end{array} \right] \in \mathbb{R}^{n \times 1} =  \sum\limits_{j=1}^d w^{\intercal}x_{i}
$$

Using this new notation we can write 

$$ \sum\limits_{i=1}^n(y_{i} - w^{\intercal}x_{i})^2 = ||y-Xw||_2^2 $$

Now we can rewrite $l(w)$ in term of the design matrix.

$$ l(w) = \frac{1}{2\sigma^2}||y-Xw||^2 + function(\sigma^2) $$

This means that the MCL (minimizing the conditional likelihood) which is the same as minimizing $\mid\mid y-Xw \mid\mid^2$

# $\hat{\omega}_{MLE}$
The resulting optimization problem is

$$ \hat{\omega}_{MLE} = \underset{w \in \mathbb{R}^d}{argmin} \, ||y-Xw||^2$$ 

To know : $\nabla_w \,(w^{\intercal}Aw) = (A + A^{\intercal})w$

$$
\begin{align*}
    0 & = \nabla_w ((y^{\intercal}-Xw)^{\intercal}(y^{\intercal}-Xw)) \\
    0 & = \nabla_w (||y||^2 - 2y^{\intercal}Xw + w^{\intercal}X^{\intercal}Xw) \\
    0 & =  0 - 2y^{\intercal}X + (X^{\intercal}X + (X^{\intercal}X)^{\intercal})w\\
    0 & = - 2y^{\intercal}X + 2X^{\intercal}Xw \\
\end{align*}
$$

This give rise to the normal equation

$$(X^{\intercal}X)w^{*} = X^{\intercal}y $$

The normal equation has 2 cases depending on weather or not $X^{\intercal}X$ is invertible.
* **Case 1 : $X^{\intercal}X$ is invertible**. This means that there is a unique solution.
For $X^{\intercal}X$  to be invertible $n \ge d$ because $rank(X) \le \min(d,n)$.The solutions are the following, here $X(X^{\intercal}X)^{-1}X^{\intercal}$ is the projection operator on the column space of X

$$\begin{align*}
\hat{w}_{MLE}  & = (X^{\intercal}X)^{-1}X^{\intercal}y \\
\hat{y} &= X\hat{w} = X(X^{\intercal}X)^{-1}X^{\intercal}y
\end{align*}$$



* **Case 1 : $X^{\intercal}X$ is NOT invertible** 
Any $\hat{w}$ such that $(X^{\intercal}X)w^{*} = X^{\intercal}y$ is a MCL estimator. It is possible to choose $\hat{w} = \underset{w:}{argmin} \, ||w|| = X^{+}y$ where  $X^{+} = (X^{\intercal}X)^{-1}X^{\intercal}$ is the Moore-Penrose pseudo-inverse, but the problem is that the pseudo-inverse is NOT numerically stable. Instead regularization is used to get a similar effect.