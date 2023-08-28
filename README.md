# kNN-and-Metric-Learning
In this repository, the kNN algorithm is being implemented. Then, using metric learning methods, including LMNN and LFDA, I am attempting to enhance the results of the kNN algorithm. Additionally, the GMML method will also be mentioned.

***kNN Classification***

**Part A:** Classifier Design

In this section, we first design the kNN (k-Nearest Neighbor) classifier using 80% of the data (136 data points). Then, we calculate the confusion matrix and classification accuracy for different values of k. The following image illustrates the confusion matrix and classification accuracy for k values of 1, 5, 10, and 20.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/629a960d-f497-49d1-9432-96e583cbcff3)


**Part B:** Calculating the Probability Distribution for Each Class

In this section, according to the problem's requirements, the Prob_knn function has been written. The following images illustrate the probability distribution for each class for k=5, 10, and 20.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/569a5f5f-a6c1-4ce5-8e24-16992778c874)

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/2dd41ce9-190b-4e9c-930f-3fd5027d83f7)

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/5e96a7e6-023a-433a-a87b-1e70c7fd4b0f)

As can be seen in the above images, for very small values of k (k=1), the accuracy is not very suitable. This is because there's a probability that a test data point is close to a non-class data point in the training set. Additionally, if k becomes too large, we might encounter errors. For instance, when the number of training data points from a specific class is low in a particular region, with a large k, a significant number of data points from the opposing class fall within the test data range, leading to errors.

***Metric Learning***

**Part A:** Examining the Performance of the Learning Method
The purpose of these two methods is to transfer data to a new space in which the data related to both distinct classes have a greater distance from each other. To achieve this goal, in each method, an optimization problem along with constraints is defined.
Initially, the Mahalanobis metric criterion is defined as follows:

(Note: Since you've only provided a portion of the text and a placeholder for the Mahalanobis metric definition, I've translated the given text and left a note indicating that there's more to the text.)

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/e65d3259-fe7d-42ca-a35e-8e5847d70ce2)

***LMNN (Large Margin Nearest Neighbor) Method:***

In this method, for defining data in a new space, we first determine the k nearest neighbors for a data point. Then, we bring the data points of the same class closer and push away the data points from different classes.

Now, we define the optimization problem as follows. (λ is a positive constant, and M is a positive definite matrix (Constraint III).)

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/c38765de-3900-44c2-9605-ef45696f7157)

Constraints (I) and (II) indicate that we intend to not only minimize the distance between similar class data points but also separate the data points of dissimilar classes.

***Local Fisher Discriminant Analysis (LFDA) Method:***

In this approach, to define the data in a new space, we attempt to assign weights to data points of the same class and consequently position them in a more independent manner. (In this method, the correlation between the features of data from two different classes becomes nearly zero.)

In this manner, we define two matrices for the optimization problem as follows.

$n_c$ represents the number of data points in class c, and $n$ represents the total number of data points.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/fd2c0b63-bd50-485a-92df-d37def384f53)

Finally, utilizing $T_{LFDA}$, the data is visualized in a new space as follows. (Where $z_i$ represents the projected image data.)

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/dc31d576-2808-44b1-b9c5-1489645143eb)

**Part B:** Visualization of Transferred Data in the New Space

- In the two discussed learning methods, `k` refers to the number of nearest neighbors to the selected data point. We aim to bring data points of the same class closer together and push data points of different classes apart. However, in the kNN classifier, we attempt to determine the class of the test data by using the majority of the classes among its k nearest neighbors.
- Initially, the original data (with 13 dimensions) is transformed into a lower-dimensional space (2 dimensions) using the PCA method. Then, employing the LMNN and LFDA learning methods, the data is transferred to a new space with even fewer dimensions (using the `n_components` argument in the LMNN or LFDA method command).

The following images depict the data plots in a 2-dimensional space using the LMNN method.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/96571c8e-3dcb-468f-a99d-cbd745738ad9)

The following images depict data plots in a 2-dimensional space using the LFDA method.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/09cb00fc-abea-46f8-b196-a126cb753a45)

As can be seen in the above images, the optimal value for k is 15 for both the LMNN and LFDA methods. This is because in this scenario, the number of decision points for approaching the selected data class or moving away from it is higher, leading to better separation between the classes.

**Part C:** Classifier Performance Comparison

In this section, we transfer the test data to a new space using the machine trained with LFDA and LMNN methods. We utilize the kNN classifier from the previous section to determine their labels.

The following image illustrates the accuracy and confusion matrix for the machine trained using the LMNN method.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/ee191529-bee1-4391-8417-d388b72a1992)

The following image illustrates the accuracy and confusion matrix for a machine trained using the LFDA method.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/e1296c51-afe8-4b96-a0ea-bb07711ee562)

As can be seen in the above images, after transferring the data to a two-dimensional space using the LFDA and LMNN learning methods, the data points are separated from each other and have a better arrangement. Consequently, the kNN classifier will perform better, and its accuracy will be significantly improved.

**Part D:** Correlation Coefficient

In this section, we first calculate the correlation matrix for both the raw and transformed data using both learning methods. To achieve this, we initially convert the data and their features into a DataFrame. Then, we utilize the `.corr()` function to obtain the correlation matrix. Finally, we visualize this matrix using the `heatmap` function from the Seaborn library.

The following image illustrates the correlation matrix for the raw data.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/2f46dafe-b0a2-47dc-8c98-c082ab7abfcb)


The following image illustrates the correlation matrix for the data transformed to the new space using the LMNN method.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/534eff43-5348-4fc5-9ce3-c34a096af840)

As seen in the above images, the values of the correlation matrix have increased after transferring the images to a new space (13-dimensional) using the LMNN method. Therefore, it can be said that this learning method enhances the separation of different classes by increasing the correlation among their features.

The following image illustrates the correlation matrix for the data transferred to the new space using the LFDA method.

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/a486fa78-539d-4285-af3a-a1e3c21b5d03)

As observed in the above images, the values of the correlation matrix after projecting the images into the new 13-dimensional space using the LFDA method are larger on the main diagonal. Consequently, with this learning approach, we have been able to somewhat independentize the features from each other.

**Part E:** GMML

In this method, an attempt is made to better represent the data in a new space by modifying the optimization problem of LMNN defined in section A. To achieve this, the optimization problem is defined as follows. (S represents data points of the same class, and D represents data points of different classes.)

![image](https://github.com/ErfanPanahi/kNN-and-Metric-Learning/assets/107314081/ead7f9b6-c0fc-4357-84e7-a9ee073f31aa)
