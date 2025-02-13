import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv("synthetic_amazon_reviews.csv")

# Combine the 'summary' and 'reviewText' columns for the word cloud
text_data = " ".join(data['summary'].dropna().astype(str) + " " + data['reviewText'].dropna().astype(str))

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)

# Line Chart - Average Rating Over Time
plt.figure(figsize=(12, 6))
data['unixReviewTime'] = pd.to_datetime(data['unixReviewTime'], unit='s')
ratings_over_time = data.groupby(data['unixReviewTime'].dt.to_period('M'))['overall'].mean()
ratings_over_time.plot()
plt.title('Average Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

# Box Plot - Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['overall'])
plt.title('Box Plot of Ratings')
plt.xlabel('Rating')
plt.show()

# Heatmap - Correlation Between Helpful Votes and Ratings
# Extracting helpful votes from the list format
data['helpful_votes'] = data['helpful'].apply(lambda x: int(x.strip('[]').split(',')[0]))
data['total_votes'] = data['helpful'].apply(lambda x: int(x.strip('[]').split(',')[1]))

correlation_data = data[['helpful_votes', 'total_votes', 'overall']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Correlations')
plt.show()

# Histogram - Distribution of Review Lengths
plt.figure(figsize=(10, 6))
data['review_length'] = data['reviewText'].apply(lambda x: len(str(x).split()))

plt.hist(data['review_length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length (words)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()