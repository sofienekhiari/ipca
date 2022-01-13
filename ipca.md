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

## Basic principles

## Analysis steps

### Import the data

```{code-cell}
:tags: ['remove-input']
# Load the data from the csv
stud_data = pd.read_csv('students_exams.csv', index_col=0)

# Display the first 5 rows of the data
display(stud_data.head(5))
```

### Run the analysis

```{code-cell}
:tags: ['remove-input']
# Standardize the data
stud_data = StandardScaler().fit_transform(stud_data)

# Run the PCA analysis
stud_pca = PCA(n_components=2).fit_transform(stud_data)

# Display the results for the first 5 rows of the data
display(pd.DataFrame(stud_pca, columns=['Principal Component 1','Principal Component 2']).head(5))
```

### Plot the results

```{code-cell}
:tags: ['remove-input']
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

# Create the plot
plt.scatter(x, y, s=25, c=colors)

# Add a title and labels
plt.title('2 component PCA', fontsize = 18)
plt.xlabel('Principal Component 1', fontsize = 14)
plt.ylabel('Principal Component 2', fontsize = 14)

# Show the plot
plt.show()
```

## Bibliography
