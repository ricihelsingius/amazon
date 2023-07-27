import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('amazon.csv')

# Cleaning the `discounted_price`, `actual_price`, and `rating_count` columns
df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)

# Dealing with missing values in `rating_count`
# Given that there are only two missing values, we can safely fill them with the median value.
df['rating_count'].fillna(df['rating_count'].median(), inplace=True)

# Replace non-numeric ratings with NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Fill NaN values with the median rating
df['rating'].fillna(df['rating'].median(), inplace=True)

# Remove duplicates from the dataframe
df = df.drop_duplicates()


# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot a simple histogram with binsize determined automatically
sns.histplot(data=df, x="discounted_price", color="skyblue", ax=axes[0, 0])
sns.histplot(data=df, x="actual_price", color="olive", ax=axes[0, 1])
sns.histplot(data=df, x="rating", color="gold", ax=axes[1, 0])
sns.histplot(data=df, x="rating_count", color="teal", ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Plot pairplot for numerical columns
sns.pairplot(df[['discounted_price', 'actual_price', 'rating', 'rating_count']])
plt.show()

# Show the top 10 categories
top_categories = df['category'].value_counts().head(10)
print(top_categories)

# Compute the correlation matrix
corr_matrix = df[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# Calculate the discount percentage
df['discount_percentage'] = (df['actual_price'] - df['discounted_price']) / df['actual_price']

# Compute the correlation between discount percentage, rating count, and rating
discount_correlation = df[['discount_percentage', 'rating_count', 'rating']].corr()

# Set up the matplotlib figure for scatter plots
f, axes = plt.subplots(1, 2, figsize=(15, 5))

# Generate scatter plots
axes[0].scatter(df['discount_percentage'], df['rating_count'])
axes[0].set_title('Discount Percentage vs Rating Count')
axes[0].set_xlabel('Discount Percentage')
axes[0].set_ylabel('Rating Count')

axes[1].scatter(df['discount_percentage'], df['rating'])
axes[1].set_title('Discount Percentage vs Rating')
axes[1].set_xlabel('Discount Percentage')
axes[1].set_ylabel('Rating')

plt.tight_layout()
plt.show()

# Return the correlation matrix
print(discount_correlation)

# Split the 'category' column
df[['main_category', 'sub_category']] = df['category'].str.split('|', n=1, expand=True)

# Display the first few rows of the dataframe
df.head()

# Set the style of the plots
sns.set_style("whitegrid")

# Create a figure
plt.figure(figsize=(15, 10))

# Create subplots
plt.subplot(2, 2, 1)
sns.boxplot(x=df['discounted_price'])
plt.title('Discounted Price')

plt.subplot(2, 2, 2)
sns.boxplot(x=df['actual_price'])
plt.title('Actual Price')

plt.subplot(2, 2, 3)
sns.boxplot(x=df['rating_count'])
plt.title('Rating Count')

plt.tight_layout()
plt.show()

# Add a small constant to handle zero values and apply a logarithmic transformation
df['log_discounted_price'] = np.log(df['discounted_price'] + 1)
df['log_actual_price'] = np.log(df['actual_price'] + 1)
df['log_rating_count'] = np.log(df['rating_count'] + 1)

# Create a figure
plt.figure(figsize=(15, 10))

# Create subplots
plt.subplot(2, 2, 1)
sns.boxplot(x=df['log_discounted_price'])
plt.title('Log Discounted Price')

plt.subplot(2, 2, 2)
sns.boxplot(x=df['log_actual_price'])
plt.title('Log Actual Price')

plt.subplot(2, 2, 3)
sns.boxplot(x=df['log_rating_count'])
plt.title('Log Rating Count')

plt.tight_layout()
plt.show()

# Count the number of products in each main category
main_category_counts = df['main_category'].value_counts()

# Count the number of products in each sub-category
sub_category_counts = df['sub_category'].value_counts()

# Calculate the average rating for each main category
main_category_avg_rating = df.groupby('main_category')['rating'].mean()

# Calculate the average rating for each sub-category
sub_category_avg_rating = df.groupby('sub_category')['rating'].mean()

# Calculate the average discount for each main category
main_category_avg_discount = df.groupby('main_category')['discount_percentage'].mean()

# Calculate the average discount for each sub-category
sub_category_avg_discount = df.groupby('sub_category')['discount_percentage'].mean()

# Plot the category analysis
plt.figure(figsize=(10, 6))
main_category_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Count of Products in Each Main Category')
plt.xlabel('Main Category')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sub_category_counts.sort_values(ascending=False)[:20].plot(kind='bar', color='olive')  # Limit to top 20 for readability
plt.title('Count of Products in Each Sub-Category (Top 20)')
plt.xlabel('Sub Category')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
main_category_avg_rating.sort_values(ascending=False).plot(kind='bar', color='gold')
plt.title('Average Rating in Each Main Category')
plt.xlabel('Main Category')
plt.ylabel('Average Rating')
plt.show()

plt.figure(figsize=(10, 6))
sub_category_avg_rating.sort_values(ascending=False)[:20].plot(kind='bar', color='teal')  # Limit to top 20 for readability
plt.title('Average Rating in Each Sub-Category (Top 20)')
plt.xlabel('Sub Category')
plt.ylabel('Average Rating')
plt.show()

plt.figure(figsize=(10, 6))
main_category_avg_discount.sort_values(ascending=False).plot(kind='bar', color='blue')
plt.title('Average Discount in Each Main Category')
plt.xlabel('Main Category')
plt.ylabel('Average Discount')
plt.show()

plt.figure(figsize=(10, 6))
sub_category_avg_discount.sort_values(ascending=False)[:20].plot(kind='bar', color='purple')  # Limit to top 20 for readability
plt.title('Average Discount in Each Sub-Category (Top 20)')
plt.xlabel('Sub Category')
plt.ylabel('Average Discount')
plt.show()
