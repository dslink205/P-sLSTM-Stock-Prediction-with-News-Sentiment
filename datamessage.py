import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the CSV file
try:
    df = pd.read_csv('1.csv')
except FileNotFoundError:
    print("Error: '1.csv' not found. Please ensure the file is in the working directory.")
    exit(1)

# Convert Date column (labeled '日期') to datetime
df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['日期'])

# Define the date range (June 1, 2021 to May 31, 2025)
start_date = datetime(2021, 6, 1)
end_date = datetime(2025, 5, 31)
total_calendar_days = (end_date - start_date).days + 1  # 1461 days

# Number of trading days (rows with valid data)
trading_days = len(df)

# Number of news articles (rows with non-null sentiment_score)
news_articles = df['情感分数'].notnull().sum()

# Mean and standard deviation of sentiment scores
sentiment_mean = df['情感分数'].mean()
sentiment_std = df['情感分数'].std()

# Mean and standard deviation of returns (Price Change %)
returns_mean = df['涨跌幅(%)'].mean()
returns_std = df['涨跌幅(%)'].std()

# Calculate missing days
# Generate all possible dates in the range
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
# Filter out weekends (Saturday=5, Sunday=6)
trading_dates = [d for d in all_dates if d.weekday() < 5]
# Approximate U.S. holidays (10 major holidays per year, ~40 over 4 years)
approx_holidays = 40
expected_trading_days = len(trading_dates) - approx_holidays  # ~1008 days
# Identify missing trading days
actual_dates = set(df['日期'])
missing_dates = [d for d in trading_dates if d not in actual_dates]
missing_days = len(missing_dates)

# Create descriptive statistics table
stats = {
    'Metric': [
        'Number of Trading Days',
        'Number of News Articles',
        'Sentiment Score Mean',
        'Sentiment Score Std Dev',
        'Returns Mean (%)',
        'Returns Std Dev (%)',
        'Missing Days'
    ],
    'Value': [
        trading_days,
        news_articles,
        f"{sentiment_mean:.4f}" if not np.isnan(sentiment_mean) else 'N/A',
        f"{sentiment_std:.4f}" if not np.isnan(sentiment_std) else 'N/A',
        f"{returns_mean:.4f}" if not np.isnan(returns_mean) else 'N/A',
        f"{returns_std:.4f}" if not np.isnan(returns_std) else 'N/A',
        missing_days
    ]
}

# Convert to DataFrame for table output
stats_df = pd.DataFrame(stats)

# Output as Markdown table
markdown_table = stats_df.to_markdown(index=False)

# Print the table for inclusion in the document
print("\nDescriptive Statistics Table for the Stock:\n")
print(markdown_table)

# Save the table to a file for inclusion in the appendix
with open('descriptive_stats_table.md', 'w') as f:
    f.write("# Descriptive Statistics Table for the Stock\n\n")
    f.write(markdown_table)
