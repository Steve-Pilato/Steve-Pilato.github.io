---
title: "Calculating the Intercept and Regression Coefficient using the Matrix Inverse in Python"
date: 2020-06-19
category: "stats"
tags: [Regression, linear algebra, data science, Python]
mathjax: "true"
---

### In linear regression, we map the predictor variable/s X to the dependent variable Y with the linear function Y = f(x) = a + bx

### You can think about this as solving a system of linear equations. For example, if we had an independent and dependent variable with n observations, we could write out the series of equations like this...

![first equation](https://latex.codecogs.com/gif.latex?%5CLARGE%20%5Cbegin%7Balign*%7D%20y_%7B1%7D%20%3D%20a%20&plus;%20b_%7B1%7Dx_%7B1%7D%20%5C%5C%20y_%7B2%7D%20%3D%20a%20&plus;%20b_%7B1%7Dx_%7B2%7D%20%5C%5C%20..............%20%5C%5C%20y_%7Bn-1%7D%20%3D%20a%20&plus;%20b_%7B1%7Dx_%7Bn-1%7D%5C%5C%20y_%7Bn%7D%20%3D%20a%20&plus;%20b_%7B1%7Dx_%7Bn%7D%20%5Cend%7Balign*%7D)

### We can change the representation of the linear system using matrix notation. 
Matrix equation: Ax = b where A is the matrix containing predictors, x is the matrix containing the coefficients, and b is the target/dependent variable.

![second equation](https://latex.codecogs.com/gif.latex?%5CLARGE%20%5Cbegin%7Bequation%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%20x_%7B1%7D%20%5C%5C%201%20%26%20x_%7B2%7D%20%5C%5C%20...%20%26%20...%20%5C%5C%201%20%26%20x_%7Bn-1%7D%5C%5C%201%20%26%20x_%7Bn%7D%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20a%5C%5C%20b_%7B1%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20y_%7B1%7D%20%5C%5C%20y_%7B2%7D%20%5C%5C%20...%20%5C%5C%20y_%7Bn-1%7D%20%5C%5C%20y_%7Bn%7D%20%5Cend%7Bbmatrix%7D%20%5Cend%7Bequation%7D)

### We can solve for x (a and b) by performing the following operations
\begin{equation}(A^{T}A)^{-1}(A^{T}A)x = (A^{T}A)^{-1}A^{T}b \ or \ x = (A^{T}A)^{-1}A^{T}b\end{equation}


## Let's import some data and use Python to test some of what we discussed 


```python
# Import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```


```python
# Import synthetic dataset 
data = pd.read_csv("C:\\Users\\steve\\OneDrive\\Documents\\website projects\\statistics\\synthetic data\\simple_regression.csv")
#View dataframe
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4.373518</td>
      <td>89.996377</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.723557</td>
      <td>96.019228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5.671688</td>
      <td>116.386921</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.860543</td>
      <td>100.563548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4.797259</td>
      <td>100.377688</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize data using matplotlib
plt.scatter(data.x, data.y, edgecolor = "black")
plt.xlabel("x - Independent Variable")
plt.ylabel("y - Dependent Variable")
```

![output](/images/regression/xvy_2.png)


### Based on the above figure, we can see that there is a clear linear relationship between x and y (What a strange coincidence)

#### Note. The numpy library has a host of linear algerbra functions (numpy.linalg)

#### Creating the A and B matracies


```python
#Create column of ones (This is for the intercept)
col1 = np.ones((data.shape[0], 1)) 
#Create a numpy array from the x variable in the above pandas data frame
col2 = np.array(data.x)
#Create A
A = np.column_stack((col1, col2))
#Create B
B = np.array(data.y)
```

#### Perform the above mentioned matrix operations


```python
#Left multiply A by A transpose
AtA = np.transpose(A) @ A
# Perform the explicit inverse on AtA
AtAinv = np.linalg.inv(AtA)
```

### Before solving for x, let's do a quick sanity check to see if the inverse was calculated correctly.
#### Remember that 
\begin{equation}(A^{T}A)^{-1}(A^{T}A) = I \ Thus,
(A^{T}A)^{-1}(A^{T}A)x = I x \end{equation}


```python
print(AtAinv @ AtA)
```

    [[1. 0.]
     [0. 1.]]
    

#### Solving for x


```python
x = AtAinv @ (np.transpose(A) @ B)
print("a = ", x[0])
print("b1 = ", x[1])
```

    a =  3.2436220896970553
    b1 =  19.9273221320193
    

#### Let's check our work by using sklearn


```python
reg = LinearRegression().fit(A, B)
print("a = ", reg.intercept_)
print("b1 =", reg.coef_[1])
```

    a =  3.2436220896965153
    b1 = 19.927322132019434
    
