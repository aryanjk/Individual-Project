import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("synthetic_amazon_reviews.csv")

# Combine the 'summary' and 'reviewText' columns for the word cloud
text_data = " ".join(data['summary'].dropna().astype(str) + " " + data['reviewText'].dropna().astype(str))

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)

# Display the word cloud
plt.figure(figsize=(15, 7.5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
