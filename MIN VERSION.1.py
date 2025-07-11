
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Importing Dataset
data = pd.read_csv('food_order_3.csv')
print("Original Dataset:")

# Ensure all columns are displayed without truncation and side by side
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Disable column wrapping
pd.set_option('display.max_colwidth', None)  # Ensure columns are not truncated

# Print dataset
print(data)

# Step 2: Check the structure of the data
print("Structure of dataset:")
data.info()

# Step 3: Change the rating dtype from object to 'float' type and replace 'Not given' with 0
data['rating'] = data['rating'].str.replace('Not given', '0', case=False)
data['rating'] = data['rating'].astype('float')

# Display the dataset after replacing 'Not given' with 0
print("Dataset after replacing 'Not given' with 0:")
print(data.head())

# Step 4: Calculate the average rating (excluding NaN values)
average_rating = data['rating'].mean()
print(f"Average Rating: {average_rating}")

# Step 5: Replace 0 ratings with the average rating
data['rating'] = np.where(data['rating'] == 0, average_rating, data['rating'])

# Display the dataset after replacing 0 values with the average rating
print("Dataset after replacing 0 ratings with the average rating:")
print(data.head())  # Display the first few rows of the updated dataset

# Step 6: Check for Duplicate Data and Drop Them
data_dup = data.duplicated().any()
print("Duplicate values present:", data_dup)
if data_dup:
    data = data.drop_duplicates()

# Step 7: Visualization Cases

# Case 1: Best 4 cuisine_type orders (American, Japanese, Italian, and Chinese)
plt.figure(figsize=(6, 4))  # Reduced figure size
sns.countplot(data['cuisine_type'], order=data['cuisine_type'].value_counts().index)
plt.xticks(rotation=70)
plt.title('Count of Orders by Cuisine Type')
plt.tight_layout()
plt.show()

# Print the top 4 cuisines with their counts
top_cuisines_counts = data['cuisine_type'].value_counts().nlargest(4)
print("Top 4 Cuisines by Order Count:")
print(top_cuisines_counts)

# Case 2: Weekend food orders compared to weekdays
plt.figure(figsize=(6, 4))  # Reduced figure size
sns.countplot(data['day_of_the_week'], order=data['day_of_the_week'].value_counts().index)
plt.xticks(rotation=70)
plt.title('Orders by Day of the Week')
plt.tight_layout()
plt.show()

# Calculate the number of weekend and weekday orders
weekend_orders = data[data['day_of_the_week'].isin(['Weekend'])].shape[0]
weekday_orders = data[data['day_of_the_week'].isin(['Weekday'])].shape[0]

print(f"Total Weekend Orders: {weekend_orders}")
print(f"Total Weekday Orders: {weekday_orders}")

# Determine which one is higher
if weekend_orders > weekday_orders:
    print("Weekend orders are higher.")
elif weekday_orders > weekend_orders:
    print("Weekday orders are higher.")
else:
    print("Weekend and weekday orders are equal.")

# Case 3: Pie chart of customer ratings
plt.figure(figsize=(6, 4))  # Reduced figure size
data['rating'].value_counts().plot(kind="pie", autopct="%1.2f%%", startangle=90, colors=sns.color_palette("Set2"))
plt.title('Customer Ratings Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Case 4: Average cost of the order by cuisine type
plt.figure(figsize=(6, 4))  # Reduced figure size
sns.barplot(x='cuisine_type', y="cost_of_the_order", data=data, order=data.groupby('cuisine_type')['cost_of_the_order'].mean().sort_values().index)
plt.xticks(rotation=50)
plt.title('Average Cost of Order by Cuisine Type')
plt.tight_layout()
plt.show()

# Case 5: Food preparation time by cuisine type
plt.figure(figsize=(6, 4))  # Reduced figure size
sns.barplot(x='cuisine_type', y="food_preparation_time", data=data, order=data.groupby('cuisine_type')['food_preparation_time'].mean().sort_values().index)
plt.xticks(rotation=50)
plt.title('Average Food Preparation Time by Cuisine Type')
plt.tight_layout()
plt.show()

# Case 6: Distribution of delivery time
plt.figure(figsize=(6, 4))  # Reduced figure size
sns.histplot(data['delivery_time'], bins=20, kde=True)
plt.title('Distribution of Delivery Time')
plt.tight_layout()
plt.show()

# Case 7: Distribution of food preparation time
plt.figure(figsize=(6, 4))  # Reduced figure size
sns.histplot(data['food_preparation_time'], bins=20, kde=True)
plt.title('Distribution of Food Preparation Time')
plt.tight_layout()
plt.show()

# Step 9: Predictive Modeling
# Prepare the data for modeling
X = data[['cost_of_the_order', 'food_preparation_time']]  # Independent variables
y = data['rating']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize predictions vs actual ratings
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', lw=2)  # Diagonal line
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.tight_layout()
plt.show()

# Step 10: Distribution of delivery time using np.histogram for better bin control
plt.figure(figsize=(6, 4))
sns.histplot(data['delivery_time'], bins=np.arange(min(data['delivery_time']), max(data['delivery_time']) + 1, 5), kde=True)
plt.title('Distribution of Delivery Time')
plt.tight_layout()
plt.show()

# Step 11: Distribution of food preparation time using np.histogram
plt.figure(figsize=(6, 4))
sns.histplot(data['food_preparation_time'], bins=np.arange(min(data['food_preparation_time']), max(data['food_preparation_time']) + 1, 5), kde=True)
plt.title('Distribution of Food Preparation Time')
plt.tight_layout()
plt.show()
