# Linear Regression Implementation From Scratch


**1. Project Overview**
> This project is a hands-on implementation of a Linear Regression model built from the ground up using Python, NumPy, and Pandas. The primary goal is to predict housing prices on the California Housing dataset without relying on high-level machine learning libraries like scikit-learn for the core modeling part.

- The model is trained using the Gradient Descent optimization algorithm, which iteratively adjusts the model's parameters (weights and bias) to minimize the Mean Squared Error (MSE) between the predicted and actual values.

- This project serves as a foundational exercise in understanding the mechanics behind one of the most fundamental machine learning algorithms.

**2. Key Features**
- From-Scratch Implementation: The core logic for prediction, loss calculation, and parameter updates is written manually using NumPy.

- *Gradient Descent*: The model's weights and bias are optimized by minimizing the cost function.

- *Mean Squared Error (MSE)*: MSE is used as the cost function to measure the model's performance during training.

- *Data Preprocessing*: Includes feature scaling using StandardScaler to ensure that Gradient Descent converges effectively.

- *Train-Test Split*: The dataset is split into training and testing sets to evaluate the model's ability to generalize to new, unseen data.

- *Visualization*: The training process is visualized by plotting the MSE loss over epochs, showing the model's convergence.

**3. Dataset**
- The model is trained and evaluated on the California Housing dataset, which is a standard regression dataset.

- Target Variable: Median house value for California districts (continuous value).

- Features: 8 numerical features including median income, house age, average number of rooms, etc.

**4. Implementation Details**
The implementation follows these key steps:

- Data Loading & Preprocessing: The dataset is loaded, and the features (X) and target (y) are separated. The features are then standardized to have a mean of 0 and a standard deviation of 1. This is crucial for the performance of Gradient Descent.

- Train-Test Split: The data is split into an 80% training set and a 20% testing set to ensure a fair evaluation of the model.

- Parameter Initialization: The model's weights (w) are initialized to a vector of zeros, and the bias (b) is initialized to zero.

**Model Training (Gradient Descent Loop)**:
The model iterates for a fixed number of epochs.
In each epoch:
a.  Prediction: It calculates the predicted y values for the training data using the current weights and bias: y_pred = X_train * w + b.
b.  Loss Calculation: It computes the MSE loss between y_pred and the actual y_train.
c.  Gradient Calculation: It calculates the partial derivatives (gradients) of the loss function with respect to the weights and bias.
d.  Parameter Update: It updates the weights and bias by taking a small step in the opposite direction of the gradient, scaled by the learning_rate.

**5. Results**
- The model was trained successfully, and the training loss was observed to decrease over time, indicating that the Gradient Descent algorithm was working correctly.

- The Training Loss Curve shows a steep drop in MSE during the initial epochs, followed by a plateau, which signifies that the model has converged to the optimal set of parameters. Training for around 400-1000 epochs was found to be sufficient for convergence.

- After training, the final learned weights and bias can be used to make predictions on the unseen test data. The model's performance is typically evaluated using metrics like R-squared or Root Mean Squared Error on the test set predictions.

**6. How to Run**
- Ensure you have a Python environment set up.

- Install the necessary dependencies.

- Place the code in a Python script (e.g., linear_regression.py).

- Run the script from your terminal:

- python linear_regression.py

**7. Dependencies**
- numpy

- pandas

- matplotlib

- scikit-learn (used only for data loading, splitting, and scaling)

# To install the dependencies, run:

**pip install numpy pandas matplotlib scikit-learn**

