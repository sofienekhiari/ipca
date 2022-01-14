---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]
# Import the needed libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
```

# Principal Component Analysis

## Abstract

## Introduction

In this essay, we will briefly introduce `Machine Learning` and the `Principal Component Analysis` in a way that is hopefully comprehensible even to readers that are not familiar with the subject. The essay comprises two main sections:

- {ref}`Basic Principles<basic-principles>`: Introduction to the basic principles of Machine Learning and `Dimensionality Reduction` with Principal Component Analysis taken as an example.
- {ref}`Analysis Conduct<analysis-conduct>`: Description of the Principal Component Analysis' steps and application thereof through a basic example.

(basic-principles)=
## Basic Principles

### Machine Learning

Machine learning is all about using and _learning_ from massive amounts of evolving data to come up with `algorithms` that help understand and solve complicated problems humanity is facing. It has many application areas.

The need for such a technology has arisen from the everlasting progress humanity made and is making that allows solving problems bearing more and more complexity. Simple algorithms aren't good enough anymore to solve such problems and the solutions needed are too complicated for us to find, mainly because of a lack of knowledge, as we often do have enough data to be able to describe such problems with good accuracy. That's why we learn using all that data and the immense computational power computers provide us with, and we're able to distill the process that can explain the data we observe and construct a good and useful `approximation` by detecting certain `patterns` or `regularities` that can help us predict the near future {cite}`Alpaydn:2010tk`.

Sometimes, the data we need to analyse is however very complicated and has a lot of `dimensions` to it. In cases like this, the techniques of Dimensionality Reduction become very handy.

### Dimensionality Reduction

Dimensionality in Machine Learning refers to the number of `attributes` or `fields` a structured dataset can have {cite}`Kumar:2021tg`. In an academic field, they can for example be a number of different exams that evaluate the knowledge of a group of students in certain domains, as we're going to see in our example analysis. Real-life complicated datasets may contain hundred of dimensions {cite}`Kumar:2021tg` and this high dimensionality may cause some issues:

- It is very challenging to plot, visualise and analyse data that has a very high dimensionality {cite}`Starmer:2018wp`, as humans are unable to visualise data beyond 3D.
- It may require a lot of computational power to analyse such data, and the result may not show good accuracy {cite}`Kumar:2021tg`.

Dimensions reduction analysis include various techniques that can lower the dimensions of data without losing key information {cite}`Kumar:2021tg`. Ideally, we should not need to perform these analysis, as the classifier should be able to use whichever features are necessary and discard the irrelevant {cite}`Alpaydn:2010tk`, but it may be preferable to perform them separately anyway for the following reasons {cite}`Alpaydn:2010tk, Kumar:2021tg`:

- They reduce the time, memory and computation needed for the training.
- They decrease the complexity of the resulting algorithm rendering it more clear and robust.
- They can be plotted and analysed visually.

There are two main methods for reducing dimensionality: `feature selection` and `feature extraction` {cite}`Alpaydn:2010tk`. Principal Component Analysis is the best known and most widely used feature extraction method and as such generally consists in finding a new set of k dimensions that are combinations of the original d dimensions {cite}`Alpaydn:2010tk`. This essay will only be focusing on the Principal Component Analysis even though there are multiple techniques that we could use.

### Principal Component Analysis

In a nutshell, Principal Component Analysis is an `unsupervised` method of Dimensionality Reduction "interested in finding a `mapping` from the inputs in the original d-dimensional space to a new (k < d)-dimensional space, with minimum loss of information" {cite}`Alpaydn:2010tk` while also eliminating the redundancy in the dataset {cite}`Kumar:2021tg`. It generally uses the `Singular Value Decomposition` {cite}`Starmer:2018wp`. During the analysis, we aim to maximise the `variance` by choosing the `eigenvector` with the largest `eigenvalue` (more details in the {ref}`Analysis Conduct<analysis-conduct>` section). The first component has the biggest variance, meaning that it holds the maximum information about our data's `clustering` potential, followed by the second component and so on. We can use the principal components we get to draw a `PCA Plot`, where we can visualise potential similar individuals cluster {cite}`Starmer:2018wp`. In the {ref}`following section<analysis-conduct>`, we will look into the different steps that we need to follow to conduct a Principal Component Analysis using a specific example.

(analysis-conduct)=
## Analysis Conduct

To better explain the analysis steps, we will choose an example. For this purpose, we're going to analyse the results of 30 students in 8 different exams and try to see if those students have homogenous results or if they can be fit into categories solely based on said results.

### Import the data

```{code-cell}
:tags: ['remove-cell']
# Load the data from the csv
stud_data = pd.read_csv('students_exams.csv', index_col=0)

# Glue the first 5 rows of the data
glue("data_table_headf", stud_data.head(5))
```

First, we start by importing the data from a `csv` file present in the same folder as our notebook. This data doesn't come from a real-life situation but was rather created for this assay purely for demonstration purposes. The {numref}`data-table-headf` shows the first five rows of the table representing the generated `dataframe`.

```{glue:figure} data_table_headf
:name: "data-table-headf"
:figwidth: 400px

First five rows of the table representing the generated `dataframe`
```

### Run the analysis

```{code-cell}
:tags: ['remove-cell']
# Standardize the data
stud_data = StandardScaler().fit_transform(stud_data)

# Run the PCA analysis
pca = PCA(n_components=2)
stud_pca = pca.fit_transform(stud_data)

# Glue the results for the first 5 rows of the data
glue("pca_table_headf", pd.DataFrame(stud_pca, columns=['Principal Component 1','Principal Component 2']).head(5).style.hide_index())

# Glue the results for the two percentages of variation for each PC
glue("pca_pc1_vp", round(pca.explained_variance_ratio_[0], 3))
glue("pca_pc2_vp", round(pca.explained_variance_ratio_[1], 3))
```

Our analysis was carried out in a single line of code using the `sklearn` library, which abstracts all the mathematical calculations for the principal component analysis' transformation of data. With this library, we only need to provide the number of principal components we want our model to have {cite}`Kumar:2021tg`. The principal component analysis watch the following steps at a high level {cite}`Starmer:2018wp,Kumar:2021tg`:

- Standardize the dataset (prerequisite for every principal component analysis).
- Compute the `covariance matrix`:
    - Calculate the average values for each variable to be able to calculate the center of the data.
    - Shift the values so that the center of the data would correspond to 0 (while keeping the proportions of the points between each other intact).
- Calculate the `Eigenvalues` and `Eigenvectors` to be able to identify principal components:
    - Fit a line through the data in a way that maximises its Eigenvalues (the sum of squared distances betweeen the projected points and the origin), while making sure that it includes the center of the data. This line is the `principal component 1`.
    - Draw a line that is perpendicular to principal component 1 and also goes through the origin. This line is the `principal component 2`.
- We use the `loading scores` (total variation) of the principal components to decide which one(s) is (are) the most important for the clustering of the data. We can use a `Scree plot` to visualise these proportions (not done in this example though). The final selection of the principal component(s) depends on how many principal components we decided we want our model to have.
- Transform the original matrix of data by using the projected points on each selected principal component. The resulting matrix is used to draw the plot as shown in the {ref}`results' plotting section<plot-the-results>`.

The StatQuest PCA step-by-step video {cite}`Starmer:2018wp` provides a very good visual representation of the different steps involved in the Principal Component Analysis.

(pca-results)=
The results of the principal component analysis in our case are displayed in {numref}`pca-table-headf`. The total variation around the principal component 1 is equal to {glue:}`pca_pc1_vp` and the one around principal component 2 to {glue:}`pca_pc2_vp`.

```{glue:figure} pca_table_headf
:name: "pca-table-headf"
:figwidth: 400px

First five rows of the generated Principal Component Analysis' results
```
(plot-the-results)=
### Plot the results

```{code-cell}
:tags: ['remove-cell']

# Create the figure
fig, ax = plt.subplots()

# Seperate the first components into different numpy arrays
x = stud_pca[:, 0]
y = stud_pca[:, 1]

# Generate the colors for the graph
colors = np.array([0])
for i in x:
    if i < -2:
        colors = np.append(colors, "darksalmon")
    elif i <= 2:
        colors = np.append(colors, "darkkhaki")
    elif i > 2:
        colors = np.append(colors, "teal")
colors = np.delete(colors, 0)

# Style the plot
plt.style.use('seaborn')
plt.tight_layout()

# Generate the plot
ax.scatter(x, y, s=25, c=colors)

# Add labels
ax.set_xlabel('Principal Component 1', fontsize = 14)
ax.set_ylabel('Principal Component 2', fontsize = 14)

# Glue the plot
glue("pca_fig", fig)
```

We can plot the results by creating a `Scatter plot` that has the principal component 1 on the x-axis and the principal component 2 on the y-axis. We can then color-code the points depending on their position on the graph relative to the principal component 1. The result is show in {numref}`pca-fig`. We can see in the resulting graph that some students cluster on the right side, some almost in the middle and some in the left side. This suggests that three categories can be made for our students solely based on their exam results, which may be helpful for further analysis. We can also see in the graph how the total variation around the principal component 1 is much higher than the one around the principal component 2 which corresponds well to the {ref}`values of total variation<pca-results>` we got from our analysis in the analysis conduct section.


```{glue:figure} pca_fig
:name: "pca-fig"

Two Components Principal Component Analysis' Scatter Plot
```

## Bibliography

```{bibliography}
```
