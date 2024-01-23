# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_regression, SequentialFeatureSelector
from sklearn.metrics import r2_score, f1_score, silhouette_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage import io
from skimage.util import img_as_float

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.set_printoptions(precision=3) 

# %% [markdown]
# # 1. Feature Selection
# 
# In the lecture, you learned about three types of feature selection. In the following, consider the F measure for a filter, Sequential Feature Selector as a wrapper, and Lasso as an embedded method.
# 
# Apply each of them to the real-world dataset from Project2 to select the two and the six most important features. Use one regressor of your choice as a baseline and the evaluation methods from the last project (including learning curves).
# 
# Utilize the $R^2$ score when reporting and analyzing your results. Also, take a look at which features are selected and which are not by the different techniques.
# 

# %%
# from project 2
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)

    # plot learning curve
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    return plt

# %%
# function to evaluate the performance and print selected features
def evaluate_features(X_selected, regressor, name):
    # split into train and tests set
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=0)

    # train the regressor
    regressor.fit(X_train, y_train)

    # predictions
    y_pred = regressor.predict(X_test)

    # evaluation with R2 
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R2-score: {r2}")

    # plot of the learning curve
    plot_learning_curve(regressor, f'Learning Curves ({name})', X_selected, y)

# %%
data = np.load('real_world.npz')

# labels
keys = list(data.keys())

labels_and_shapes = {key: data[key].shape for key in keys}

for label, shape in labels_and_shapes.items():
    print(f"Label: {label}, Shape: {shape}")

X = data['X']
y = data['y']
features = data['features']

print(features)

# check for NaN
print("NaN values in X: ", np.isnan(X).sum())
print("NaN values in y: ", np.isnan(y).sum())

print("\n\n")
for i in range(X.shape[1]):
    # print(X[:,i])
    print(f"Mean of X{i}: ", np.mean(X[:,i]))
    print(f"Standard deviation of X{i}: ", np.std(X[:,i]))
    print(f"Min of X{i}: ", np.min(X[:,i]))
    print(f"Max of X{i}: ", np.max(X[:,i]))
print("\n\n")

# statistical description
print("Mean of X: ", np.mean(X))
print("Standard deviation of X: ", np.std(X))
print("Min of X: ", np.min(X))
print("Max of X: ", np.max(X))

print("Mean of y: ", np.mean(y))
print("Standard deviation of y: ", np.std(y))
print("Min of y: ", np.min(y))
print("Max of y: ", np.max(y))

# %%
# Random forest as baseline regressor
baseline_regressor = RandomForestRegressor(random_state=0)

# F measure as filter method

f_values, _ = f_regression(X, y)
selected_features_filter = X[:, np.argsort(f_values)[-2:]]  # 2 most important features
evaluate_features(selected_features_filter, baseline_regressor, 'F measure (2 features)')
print("2 selected Features with F measure:")
print(features[np.argsort(f_values)[-2:]])

selected_features_filter = X[:, np.argsort(f_values)[-6:]]  # 6 most important features
evaluate_features(selected_features_filter, baseline_regressor, 'F measure (6 features)')
print("6 selected Features with F measure:")
print(features[np.argsort(f_values)[-6:]])


# Sequential feature selector as wrapper method 

# Forward selection 
sfs = SequentialFeatureSelector(baseline_regressor, n_features_to_select=2, direction='forward', cv=5)
selected_features_wrapper = sfs.fit_transform(X, y)
evaluate_features(selected_features_wrapper, baseline_regressor, 'Forward selection (2 features)')
print("2 selected Features with forward selection:")
print(features[np.where(sfs.get_support())[0]])


sfs = SequentialFeatureSelector(baseline_regressor, n_features_to_select=6, direction='forward', cv=5)
selected_features_wrapper = sfs.fit_transform(X, y)
evaluate_features(selected_features_wrapper, baseline_regressor, 'Forward selection (6 features)')
print("6 selected Features with forward selection:")
print(features[np.where(sfs.get_support())[0]])

# Backward selection
sfs = SequentialFeatureSelector(baseline_regressor, n_features_to_select=2, direction='backward', cv=5)
selected_features_wrapper = sfs.fit_transform(X, y)
evaluate_features(selected_features_wrapper, baseline_regressor, 'Backward selection (2 features)')
print("2 selected Features with backward selection:")
print(features[np.where(sfs.get_support())[0]])

sfs = SequentialFeatureSelector(baseline_regressor, n_features_to_select=6, direction='backward', cv=5)
selected_features_wrapper = sfs.fit_transform(X, y)
evaluate_features(selected_features_wrapper, baseline_regressor, 'Backward selection (6 features)')
print("6 selected Features with backward selection:")
print(features[np.where(sfs.get_support())[0]])

# %%
# Perform Lasso with a specified number of features
def lasso_feature_selection(X, y, num_features, alpha=0.1):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)

    # Indices of the 'num_features' most important features
    selected_feature_indices = np.argsort(np.abs(lasso.coef_))[::-1][:num_features]

    # Selecting the corresponding features
    selected_features_lasso = X[:, selected_feature_indices]
    
    return selected_features_lasso, selected_feature_indices

num_features_2 = 2
num_features_6 = 6

# Lasso with 2 features
selected_features_lasso_2, indices_lasso_2 = lasso_feature_selection(X, y, num_features_2)
evaluate_features(selected_features_lasso_2, baseline_regressor, f'Embedded Method (Lasso, {num_features_2} features)')
print(f"Selected Features for Lasso (2 features): {features[indices_lasso_2]}")

# Lasso with 6 features
selected_features_lasso_6, indices_lasso_6 = lasso_feature_selection(X, y, num_features_6)
evaluate_features(selected_features_lasso_6, baseline_regressor, f'Embedded Method (Lasso, {num_features_6} features)')
print(f"Selected Features for Lasso (6 features): {features[indices_lasso_6]}")


# %% [markdown]
# # 2. Random Forest and Feature Importances
# 
# (a) Train and evaluate with cross-validation a random forest classifier, and the other classifiers you know from the lecture on `dataset1.npz` (Naive Bayes, Logistic Regression, kNN).

# %%
data = np.load('dataset1.npz')

# labels
keys = list(data.keys())

labels_and_shapes = {key: data[key].shape for key in keys}

for label, shape in labels_and_shapes.items():
    print(f"Label: {label}, Shape: {shape}")

X = data['X']
y = data['y']

# check for NaN
print("NaN values in X: ", np.isnan(X).sum())
print("NaN values in y: ", np.isnan(y).sum())

print("\n\n")
for i in range(X.shape[1]):
    # print(X[:,i])
    print(f"Mean of X{i}: ", np.mean(X[:,i]))
    print(f"Standard deviation of X{i}: ", np.std(X[:,i]))
    print(f"Min of X{i}: ", np.min(X[:,i]))
    print(f"Max of X{i}: ", np.max(X[:,i]))
print("\n\n")

# statistical description
print("Mean of X: ", np.mean(X))
print("Standard deviation of X: ", np.std(X))
print("Min of X: ", np.min(X))
print("Max of X: ", np.max(X))

print("Mean of y: ", np.mean(y))
print("Standard deviation of y: ", np.std(y))
print("Min of y: ", np.min(y))
print("Max of y: ", np.max(y))

# split into train and tests set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# %%
# Random forest classifier
rf_classifier = RandomForestClassifier(random_state=0)
rf_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy')
print(f"Random forest classifier accuracy: {np.mean(rf_scores)}")

# Naive Bayes classifier (Gaussian Naive Bayes)
nb_classifier = GaussianNB()
nb_scores = cross_val_score(nb_classifier, X_train, y_train, cv=5, scoring='accuracy')
print(f"Naive Bayes classifier Caccuracy: {np.mean(nb_scores)}")

# Logistic regression classifier
lr_classifier = LogisticRegression(random_state=0)
lr_scores = cross_val_score(lr_classifier, X_train, y_train, cv=5, scoring='accuracy')
print(f"Logistic regression classifier accuracy: {np.mean(lr_scores)}")

# k-nearest neighbors classifier 
knn_classifier = KNeighborsClassifier()
knn_scores = cross_val_score(knn_classifier, X_train, y_train, cv=5, scoring='accuracy')
print(f"k-Nearest neighbors classifier accuracy: {np.mean(knn_scores)}")

# %% [markdown]
# (b) Visualize the data by plotting each combination of two features. Analyze the feature importances of the random forest with respect to the data. Rerun your experiments on a suitable subset of the features.

# %%
# Plot each combination of two features and save individual plots
def plot_feature_combinations(X, y, save_folder):
    n_features = X.shape[1]

    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Paired, edgecolors='k', s=20)
                ax.set_xticks(())
                ax.set_yticks(())
                ax.set_xlabel(f'Feature {i}')
                ax.set_ylabel(f'Feature {j}')
                plt.title(f'Feature Combination {i} vs {j}')

                # Save the individual plot to the specified folder
                save_path = os.path.join(save_folder, f'feature_combination_{i}_vs_{j}.png')
                plt.savefig(save_path)
                plt.close()

# %%
output_folder = 'feature_plots'
os.makedirs(output_folder, exist_ok=True)

# Visualize data by plotting each combination of two features and save individual plots
plot_feature_combinations(X, y, save_folder=output_folder)

# Train Random Forest and analyze feature importances
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X, y)

# Plot feature importances
feature_importances = rf_classifier.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
features = [f'Feature {i}' for i in sorted_idx]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances[sorted_idx], align="center")
plt.xticks(range(X.shape[1]), features, rotation=45)
plt.title("Random Forest Feature Importances")
plt.show()

# Rerun experiments on a suitable subset of features
k = 5
X_subset = X[:, sorted_idx[:k]]

# Rerun experiments with the subset of features
rf_classifier_subset = RandomForestClassifier(random_state=0)
rf_scores_subset = cross_val_score(rf_classifier_subset, X_subset, y, cv=5, scoring='accuracy')
print(f"Random Forest Classifier (Subset) Cross-Validation Accuracy: {np.mean(rf_scores_subset)}")


# %%
r2s = []
baseline_regressor = RandomForestRegressor(random_state=0)
best_r2 = -float('inf')
best_selected_features = None

for i in range(1, X.shape[1]):
    sfs = SequentialFeatureSelector(baseline_regressor, n_features_to_select=i, direction='forward', cv=5)
    selected_features_wrapper = sfs.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(selected_features_wrapper, y, test_size=0.2, random_state=0)
    baseline_regressor.fit(X_train, y_train)
    y_pred = baseline_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2s.append({'r2': r2, 'selected_features': sfs.get_support().tolist()})

    # Update best R2 and selected features if current R2 is better
    if r2 > best_r2:
        best_r2 = r2
        best_selected_features = sfs.get_support().tolist() # mask

print("Best R-squared:", best_r2)
print("Corresponding selected features:", best_selected_features)

# %%
# Selected columns
X_new = X[:, best_selected_features]
X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=0)

# Random forest classifier with feature selection
rf_classifier = RandomForestClassifier(random_state=0)
rf_scores = cross_val_score(rf_classifier, X_train_new, y_train, cv=5, scoring='accuracy')
print(f"Random Forest Classifier Cross-Validation Accuracy on a subset of features: {np.mean(rf_scores)}")

# Naive Bayes classifier (Gaussian Naive Bayes) with feature selection
nb_classifier = GaussianNB()
nb_scores = cross_val_score(nb_classifier, X_train_new, y_train, cv=5, scoring='accuracy')
print(f"Naive Bayes Classifier Cross-Validation Accuracy on a subset of features: {np.mean(nb_scores)}")

# Logistic regression classifier with feature selection
lr_classifier = LogisticRegression(random_state=0)
lr_scores = cross_val_score(lr_classifier, X_train_new, y_train, cv=5, scoring='accuracy')
print(f"Logistic Regression Classifier Cross-Validation Accuracy on a subset of features: {np.mean(lr_scores)}")

# k-nearest neighbors classifier with feature selection
knn_classifier = KNeighborsClassifier()
knn_scores = cross_val_score(knn_classifier, X_train_new, y_train, cv=5, scoring='accuracy')
print(f"k-Nearest Neighbors Classifier Cross-Validation Accuracy on a subset of features: {np.mean(knn_scores)}")

# %% [markdown]
# # 3. Challenge
# 
# This exercise constitutes a challenge: Apply the learned concepts of the lecture and potentially think of new solutions in order to achieve an F1 score (use class 1 as 'positive' class) of at least 0.95 for the test set of `dataset2.npz`, without using it for training.
# 

# %%
data = np.load('dataset2.npz')

# labels
keys = list(data.keys())

labels_and_shapes = {key: data[key].shape for key in keys}

for label, shape in labels_and_shapes.items():
    print(f"Label: {label}, Shape: {shape}")

X_train = data['X_train']
y_train = data['Y_train']
X_test = data['X_test']
y_test = data['Y_test']

# check for NaN
print("NaN values in X_train: ", np.isnan(X_train).sum())
print("NaN values in y_train: ", np.isnan(y_train).sum())
print("NaN values in X_test: ", np.isnan(X_test).sum())
print("NaN values in y_test: ", np.isnan(y_test).sum())

print("\n\n")
for i in range(X_train.shape[1]):
    # print(X_train[:,i])
    print(f"Mean of X_train{i}: ", np.mean(X_train[:,i]))
    print(f"Standard deviation of X_train{i}: ", np.std(X_train[:,i]))
    print(f"Min of X_train{i}: ", np.min(X_train[:,i]))
    print(f"Max of X_train{i}: ", np.max(X_train[:,i]))
print("\n\n")

print("\n\n")
for i in range(X_test.shape[1]):
    # print(X_test[:,i])
    print(f"Mean of X_test{i}: ", np.mean(X_test[:,i]))
    print(f"Standard deviation of X_test{i}: ", np.std(X_test[:,i]))
    print(f"Min of X_test{i}: ", np.min(X_test[:,i]))
    print(f"Max of X_test{i}: ", np.max(X_test[:,i]))
print("\n\n")

# statistical description
print("Mean of X_train: ", np.mean(X_train))
print("Standard deviation of X_train: ", np.std(X_train))
print("Min of X_train: ", np.min(X_train))
print("Max of X_train: ", np.max(X_train))

print("Mean of X_test: ", np.mean(X_test))
print("Standard deviation of X_test: ", np.std(X_test))
print("Min of X_test: ", np.min(X_test))
print("Max of X_test: ", np.max(X_test))

print("Mean of y_train: ", np.mean(y_train))
print("Standard deviation of y_train: ", np.std(y_train))
print("Min of y_train: ", np.min(y_train))
print("Max of y_train: ", np.max(y_train))

print("Mean of y_test: ", np.mean(y_test))
print("Standard deviation of y_test: ", np.std(y_test))
print("Min of y_test: ", np.min(y_test))
print("Max of y_test: ", np.max(y_test))

# %%
# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_f1_score = f1_score(y_test, nb_predictions, pos_label=1)

# Logistic Regression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)
lr_f1_score = f1_score(y_test, lr_predictions, pos_label=1)

# kNN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_f1_score = f1_score(y_test, knn_predictions, pos_label=1)

# RandomForest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_f1_score = f1_score(y_test, rf_predictions, pos_label=1)

# Print results
print("Naive Bayes F1 Score:", nb_f1_score)
print("Logistic Regression F1 Score:", lr_f1_score)
print("kNN F1 Score:", knn_f1_score)
print("RandomForest F1 Score:", rf_f1_score)

# %%
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)
nb_predictions = nb_classifier.predict(X_test_scaled)
nb_f1_score = f1_score(y_test, nb_predictions, pos_label=1)

# Logistic Regression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_scaled, y_train)
lr_predictions = lr_classifier.predict(X_test_scaled)
lr_f1_score = f1_score(y_test, lr_predictions, pos_label=1)

# RandomForest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_scaled, y_train)
rf_predictions = rf_classifier.predict(X_test_scaled)
rf_f1_score = f1_score(y_test, rf_predictions, pos_label=1)

# kNN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_scaled, y_train)
knn_predictions = knn_classifier.predict(X_test_scaled)
knn_f1_score = f1_score(y_test, knn_predictions, pos_label=1)

print("Naive Bayes F1 Score:", nb_f1_score)
print("Logistic Regression F1 Score:", lr_f1_score)
print("kNN F1 Score:", knn_f1_score)
print("RandomForest F1 Score:", rf_f1_score)


# %% [markdown]
# # 4. Bonus Task - Clustering
# 
# Load the three given pictures and compress them by clustering the pixels of each picture according to their color values. Then, represent each pixel of a cluster in the mean color of the cluster. Visualize your results for different reasonable numbers of clusters using at least one evaluation score.
# 

# %%
# function to perform k-means clustering on an image
def cluster_image(image, n_clusters):
    # Reshape the image to a 2D array of pixels
    pixels = img_as_float(image).reshape((-1, 3))

    # k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # assigning new colors to pixels based on cluster centers
    compressed_image = centers[labels].reshape(image.shape)

    return compressed_image, kmeans  # Return kmeans object for later use

# function to visualize the compressed image
def visualize_compressed_image(original_image, compressed_image, n_clusters):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')

    ax[1].imshow(compressed_image)
    ax[1].set_title(f'Compressed Image ({n_clusters} clusters)')

    plt.show()

# Load the images
image_paths = ['img/CITEC.png', 'img/UBI_X.png', 'img/UHG.png']
images = [io.imread(path) for path in image_paths]

# Choose the number of clusters for each image
n_clusters_list = [[2, 4, 8], [3, 5, 7], [4, 6, 9]]

for image, n_clusters_for_image in zip(images, n_clusters_list):
    for n_clusters in n_clusters_for_image:
        compressed_image, kmeans = cluster_image(image, n_clusters)
        visualize_compressed_image(image, compressed_image, n_clusters)

        # silhouette score for evaluation
        silhouette_avg = silhouette_score(img_as_float(image).reshape((-1, 3)), kmeans.labels_)
        print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")


