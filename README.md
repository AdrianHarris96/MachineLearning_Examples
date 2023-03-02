# MachineLearning_Examples README
Various Machine Learning Scenarios/Approaches 

Regressions
- `sklearn_basics.py`: Basics of Scikit-learn (generating linear, cluster and circular data)
- `linear_regression.py`: Fitting linear model to the boston house-price data in both 2D and 3D space. Making predictions and evaluating the mean squared error between the test and prediction data. 
- `polynomial_regression.py`: Fitting polynomial (2nd and 3rd degree) to the boston house-price data in 3D space. 

Classifications 
- `log_regression_classification.py `: Fitting breast cancer data to a logistic regression model to delineate benign tumors from malignant tumors. Fit using either single features or all features and consequently, make predictions using the test set. 
- `svm_classification.py  `: Classifying iris dataset using different SVM kernels (linear, rbf, polynomial)
- `knn_classification.py  `: Classify iris dataset using K-Nearest Neighbors.

Clustering 
- `k_means_clustering  `: Generate clusters to formulate clothing sizes using points with metrics for body dimensions. Calculate silhoutte score for multiple ks to determine the optimal k. 
- `k_means_manual_clustering.py  `: Manual implementation of KMeans clustering algorithm. Randomly generate centroids, create new centroids by taking the mean, calculate the distance between old and new centroids until the distance is 0, indicating centroids are in the optimal location.