
# Linear Regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It is one of the simplest and most commonly used techniques in machine learning and data analysis.

## Key Concepts

- **Dependent Variable (Y):** The outcome or target variable that we aim to predict or explain.
- **Independent Variable(s) (X):** The input features or predictors used to model the dependent variable.
- **Linear Relationship:** Linear regression assumes a linear relationship between the dependent and independent variables.

## Equation

The equation for simple linear regression (with one independent variable) is:

```
Y = β₀ + β₁X + ε
```

Where:
- `Y` is the predicted value.
- `β₀` is the intercept (value of Y when X = 0).
- `β₁` is the slope (rate of change of Y with respect to X).
- `X` is the independent variable.
- `ε` is the error term (difference between actual and predicted values).

For multiple linear regression (with multiple independent variables), the equation becomes:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βnXn + ε
```

## Applications

- Predicting house prices based on features like size, location, etc.
- Estimating sales based on advertising spend.
- Modeling relationships in scientific research.

## Assumptions

1. Linearity: The relationship between X and Y is linear.
2. Independence: Observations are independent of each other.
3. Homoscedasticity: Constant variance of errors.
4. Normality: Errors are normally distributed.

Linear regression is a foundational technique that serves as a building block for more advanced machine learning models.


## Gradient Descent in Linear Regression

Gradient Descent is an optimization algorithm used to minimize the cost function in linear regression by iteratively updating the model parameters. It helps find the best-fit line by reducing the error between predicted and actual values.

### Cost Function

The cost function for linear regression is the Mean Squared Error (MSE):

```
J(β₀, β₁) = (1/2m) * Σ [h(Xᵢ) - Yᵢ]²
```

![alt text](image.png)

Where:
- `m` is the number of training examples.
- `h(Xᵢ)` is the predicted value: `h(Xᵢ) = β₀ + β₁Xᵢ`.
- `Yᵢ` is the actual value.

### Gradient Descent Algorithm

The goal is to minimize `J(β₀, β₁)` by updating the parameters `β₀` and `β₁` iteratively:

1. Initialize `β₀` and `β₁` with random values.
2. Update the parameters using the following equations:

```
β₀ := β₀ - α * (∂J/∂β₀)
β₁ := β₁ - α * (∂J/∂β₁)
```
![alt text](image-2.png)


Where:
- `α` is the learning rate (controls the step size).
- `∂J/∂β₀` and `∂J/∂β₁` are the partial derivatives of the cost function.

### Derivation of Gradients

The partial derivatives of the cost function are:

1. For `β₀`:
```
∂J/∂β₀ = (1/m) * Σ [h(Xᵢ) - Yᵢ]
```

2. For `β₁`:
```
∂J/∂β₁ = (1/m) * Σ [h(Xᵢ) - Yᵢ] * Xᵢ
```


![alt text](image-1.png)
Substitute these gradients into the update rules to iteratively adjust `β₀` and `β₁`.

### Iterative Process

Repeat the updates until the cost function converges (i.e., changes in `J(β₀, β₁)` become negligible). The final values of `β₀` and `β₁` represent the optimal parameters for the linear regression model.

### Visualization

Gradient Descent can be visualized as moving downhill on a surface (cost function) to reach the lowest point (minimum error).


