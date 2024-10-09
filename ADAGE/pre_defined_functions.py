import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def mean(df: pd.DataFrame, column: str) -> float:
    """Calculate the mean of a specified column in the DataFrame."""
    total = sum(df[column])
    num = len(df[column])
    return total / num

def median(df: pd.DataFrame, column: str) -> float:
    """Calculate the median of a specified column in the DataFrame."""
    sorted_values = sorted(df[column])
    mid_index = len(sorted_values) // 2
    # Return median based on odd or even number of observations
    if len(sorted_values) % 2 == 0:
        return (sorted_values[mid_index - 1] + sorted_values[mid_index]) / 2
    return sorted_values[mid_index]

def lineplot(df: pd.DataFrame, column: str, column2: str) -> None:
    """Create a line plot for the trend of a specified column over time."""
    df = df[(df['JOBNUM'] == 342135144) & (df['ROWNUM'] == 11)]
    plt.plot(df['SLINUM'], df[column], label=column, color='blue')
    plt.plot(df['SLINUM'], df[column2], label=column2, color='green')

    # Adding title and labels
    plt.title('Sales and Profit Over SLINUM')
    plt.xlabel('SLINUM')
    plt.ylabel('Value')

    # Adding grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def pie_chart(df: pd.DataFrame, column: str) -> None:
    """Create a pie chart for the distribution of values in a specified column."""
    counts = df[column].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribution of {column.capitalize()}')
    plt.axis('equal')
    plt.show()

def bar_graph(df: pd.DataFrame, column: str) -> None:
    """Create a bar graph for the distribution of values in a specified column."""
    counts = df[column].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(counts.index, counts.values, color='skyblue')
    plt.xlabel(column.capitalize())
    plt.ylabel('Counts')
    plt.title(f'Distribution of {column.capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def correlation(df: pd.DataFrame, column1: str, column2: str) -> str:
    """Calculate the Pearson correlation coefficient between two specified columns and provide its interpretation."""
    # Ensure the columns exist in the DataFrame

    print(f'{column1.capitalize()} {column2.capitalize()}')
    if column1 not in df.columns or column2 not in df.columns:
        raise ValueError(f"Columns '{column1}' or '{column2}' not found in DataFrame.")

    # Calculate necessary sums and counts
    n = len(df)
    sum_x = sum(df[column1])
    sum_y = sum(df[column2])
    sum_x_squared = sum(df[column1] ** 2)
    sum_y_squared = sum(df[column2] ** 2)
    sum_xy = sum(df[column1] * df[column2])

    # Calculate the correlation coefficient
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = ((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2)) ** 0.5

    if denominator == 0:
        return "No correlation (division by zero)."

    correlation_value = numerator / denominator

    # Provide interpretation based on correlation value
    if correlation_value == 0:
        meaning = "No correlation."
    elif -0.1 < correlation_value < 0.1:
        meaning = "Very weak or no correlation."
    elif -0.3 < correlation_value <= -0.1:
        meaning = "Weak negative correlation."
    elif -0.5 < correlation_value <= -0.3:
        meaning = "Moderate negative correlation."
    elif -0.7 < correlation_value <= -0.5:
        meaning = "Strong negative correlation."
    elif correlation_value <= -0.7:
        meaning = "Very strong negative correlation."
    elif 0.1 < correlation_value < 0.3:
        meaning = "Weak positive correlation."
    elif 0.3 <= correlation_value < 0.5:
        meaning = "Moderate positive correlation."
    elif 0.5 <= correlation_value < 0.7:
        meaning = "Strong positive correlation."
    else:
        meaning = "Very strong positive correlation."

    return f"Pearson Correlation: {correlation_value:.2f} ({meaning})"



def standard_deviation(data, column):
    """Computes the standard deviation of a dataset and visualizes a bell curve."""
    data = data[column]
    n = len(data)
    if n == 0:
        return 0.0
    
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    
    # Visualization of Bell Curve
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)  # Generate x values
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)  # Normal distribution

    plt.figure(figsize=(10, 5))
    plt.title(f'{column}')
    plt.plot(x, y, color='blue', label='Bell Curve')
    plt.axvline(mean, color='r', linestyle='--', label='Mean')
    plt.axvline(mean + std_dev, color='g', linestyle='--', label='Mean + 1 Std Dev')
    plt.axvline(mean - std_dev, color='g', linestyle='--', label='Mean - 1 Std Dev')
    plt.fill_between(x, y, alpha=0.1, color='blue')

    # Annotating the standard deviation on the plot
    plt.text(mean, max(y) * 0.7, f'Standard Deviation: {std_dev:.2f}', 
             horizontalalignment='center', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Data Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()
    
    return std_dev

def histogram_plot(data, column):
    plt.hist(data[column], bins=10, color='orange', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()