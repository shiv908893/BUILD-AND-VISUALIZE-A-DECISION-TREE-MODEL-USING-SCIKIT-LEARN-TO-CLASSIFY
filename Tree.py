import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv('car_sales.xls - car_sales.csv')

# 2. Select Features and Target
# We use numeric columns to predict the 'Origin' of the vehicle
features = [
    'Price', 'MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 
    'Rev.per.mile', 'Fuel.tank.capacity', 'Passengers', 'Length', 
    'Wheelbase', 'Width', 'Turn.circle', 'Weight'
]
X = df[features]
y = df['Origin']

# 3. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Initialize and Train the Decision Tree
# max_depth is limited to 3 to ensure the visualization remains readable and avoids overfitting
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Visualization
plt.figure(figsize=(16, 8))
plot_tree(
    clf, 
    feature_names=features, 
    class_names=clf.classes_, 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title("Decision Tree: Classifying Car Origin (USA vs. non-USA)")
plt.savefig('decision_tree.png')
plt.show()
