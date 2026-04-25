# TikTok Performance Analysis from Indonesian News Media
TikTok is a key platform for news, with Indonesian media competing in a fast-paced content environment. Understanding content, timing, and posting behavior is essential to optimize reach and engagement. This project aims to analyze TikTok data to generate insights, build a simple dashboard for data visualization, and provide data-driven recommendations to improve content strategy.

**Disclaimer**: this analysis is still at a **basic level** and may contain errors or inaccuracies. So, further validation are recommended.

## Dataset
**TikTok Performance Data**

The data is downloaded from [SocialInsider](https://www.socialinsider.io/) and covers TikTok performance from 7 Indonesian news media over a 30-day period (22 March–20 April 2026).

This data consists of these columns:

- Profile: the name of the TikTok account.
- Profile Link: URL to the TikTok profile.
- Post Link: URL to the specific post.
- Text: caption or text content of the post.
- Date: date when the post was published (YYYY-MM-DD HH:MM:SS).
- Image Link: link to the thumbnail.
- Type: type of content (video, image, etc.).
- Video Length (seconds): duration of the video in seconds.
- Engagement: total engagement (likes, comments, shares, saves).
- Likes: number of likes received.
- Shares: number of shares received.
- Saves: number of saves.
- Comments: number of comments received.
- Views: total number of views.
- Engagement Rate By Followers: engagement rate based on followers.
- Engagement Rate By Views: engagement rate based on views.
- Music Name: name of the audio used.
- Music Author: creator of the audio.
- Hashtags: hashtags used in the post.
- Brand Content Pillars: content category based on brand strategy.
- Industry Content Pillars: content category based on industry classification.
- Organic Value: estimated value of organic performance.
- Country: country of the audience or posting origin.

## Setup
### Data Loading
```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load and read data
df = pd.read_csv("FILE_NAME.csv")
df.head()
```
There are 4,275 rows and 23 columns.

### Data Cleaning
```python
# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Check duplicate row
print("\nTotal duplicate rows:")
print(df.duplicated().sum())
```
There are 4 columns with 100% missing values, 2 columns with partial and minor missing values, and no duplicate rows found. All columns with missing values are not relevant to the analysis and will be removed from data.

```python
# Convert datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert numeric columns
num_cols = ['Video Length (seconds)','Views','Engagement','Likes','Shares','Saves','Comments','Engagement Rate By Followers', 'Engagement Rate By Views']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in specified metric columns
metrics_to_check_na = ['Video Length (seconds)', 'Engagement', 'Likes', 'Shares', 'Saves', 'Comments', 'Views', 'Engagement Rate By Followers', 'Engagement Rate By Views']
df = df.dropna(subset=metrics_to_check_na)

# Drop unused columns
columns_to_drop = ['Music Name', 'Music Author', 'Hashtags', 'Brand Content Pillars', 'Industry Content Pillars', 'Organic Value', 'Country', 'Type']
df = df.drop(columns=columns_to_drop)

df.head()
```
The cleaned data now is 4,275 rows and 15 columns. After cleaning the data, some of EDA will be conducted in Python. The process will include generating some visualizations for analysis, such as views and engagement distribution and heatmap correlation. Then, the rest of EDA will use dashboard visualization.

### Data Analysis
```python
# Global font setting
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

bar_color = "#FE2C55"
line_color = "#25F4EE"

# Views distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Views'], bins=50, density=True, color=bar_color, edgecolor='black', alpha=0.7)
sns.kdeplot(df['Views'], color=line_color, linewidth=2)
plt.title('Distribution of Views')
plt.xlabel('Views')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Engagement distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Engagement'], bins=50, density=True, color=bar_color, edgecolor='black', alpha=0.7)
sns.kdeplot(df['Engagement'], color=line_color, linewidth=2)
plt.title('Distribution of Engagement')
plt.xlabel('Engagement')
plt.ylabel('Density')
plt.tight_layout()
plt.show()
```
- The distributions of both views and engagement are heavily right-skewed, indicating that performance is concentrated in a small number of high-performing posts rather than being evenly distributed.
- Given this skewness, the median provides a more reliable measure, as it minimizes distortion from viral outliers and better represents the typical content performance.

```python
# Correlation Heatmap

# Aggregate data by date to get daily total views, engagement, and number of posts
daily_aggregated_df = df.groupby(df['Date'].dt.date).agg(
    total_engagement=('Engagement', 'sum'),
    total_likes=('Likes', 'sum'),
    total_shares=('Shares', 'sum'),
    total_saves=('Saves', 'sum'),
    total_comments=('Comments', 'sum'),
    total_views=('Views', 'sum'),
    avg_engagement_rate_by_followers=('Engagement Rate By Followers', 'mean'),
    avg_engagement_rate_by_views=('Engagement Rate By Views', 'mean'),
    number_of_posts=('Post Link', 'size')
).reset_index()

daily_aggregated_df['Date'] = pd.to_datetime(daily_aggregated_df['Date'])

# Create full date range (22 Mar–20 Apr) and reindex to ensure all dates are present
date_range_agg = pd.date_range(start='2026-03-22', end='2026-04-20')
daily_aggregated_df = daily_aggregated_df.set_index('Date').reindex(date_range_agg).fillna(0).rename_axis('Date').reset_index()

# Rename columns for cleaner display in the heatmap
daily_aggregated_df = daily_aggregated_df.rename(columns={
    'total_engagement': 'Total Engagement',
    'total_likes': 'Total Likes',
    'total_shares': 'Total Shares',
    'total_saves': 'Total Saves',
    'total_comments': 'Total Comments',
    'total_views': 'Total Views',
    'avg_engagement_rate_by_followers': 'Avg Eng. Rate by Followers',
    'avg_engagement_rate_by_views': 'Avg Eng. Rate by Views',
    'number_of_posts': 'Number of Posts'
})

# Select numerical columns for correlation calculation from the aggregated data
numerical_columns_aggregated = [
    'Total Engagement', 'Total Likes', 'Total Shares', 'Total Saves',
    'Total Comments', 'Total Views', 'Avg Eng. Rate by Followers',
    'Avg Eng. Rate by Views', 'Number of Posts'
]

correlation_matrix = daily_aggregated_df[numerical_columns_aggregated].corr()

# Define custom colormap
min_color_hex = '#25F4EE' # Cyan
max_color_hex = '#FE2C55' # Red/Pink
neutral_color_hex = '#FFFFFF' # White

cmap = LinearSegmentedColormap.from_list('custom_cmap', [min_color_hex, neutral_color_hex, max_color_hex])

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap=cmap,
    fmt=".2f",
    linewidths=.5,
    linecolor='white'
)
# plt.title('Correlation Heatmap of Daily Aggregated Metrics')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels
plt.tight_layout()
plt.show()
```
- Total Views and Total Engagement showed a very strong positive correlation (0.91), suggesting that higher visibility content consistently translated into higher engagement.
Among engagement metrics, Total Saves (0.93) and Total Likes (0.89) were the strongest correlated with Views, indicating they were the most reliable indicators of content success.
- Number of Posts had a moderate correlation with Total Views (0.67) and Total Engagement (0.53), implying that posting frequency contributed to reach and engagement, but was not the primary performance driver.

The cleaned dataset will be exported as a CSV file. The file will be used to build dashboard in Google Data Studio, then the rest of EDA will be conducted using this dashboard.

Link to the dashboard is [here](https://datastudio.google.com/u/0/reporting/3973755b-b19c-40c5-aaa8-f70674b2800b).
