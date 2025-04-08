import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data
data = {
    "Hours_Studied": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10],
    "Passed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
data = pd.DataFrame(data)
x = data['Hours_Studied']
y = data['Passed'].astype(float)
y = y.to_numpy()

# adding the bias
x = np.c_[np.ones(x.shape[0]) , x ]
# weights
w = np.random.rand(2)

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cost function
def CostFunction(x , y , w):
    n = len(y)
    predict = sigmoid(np.dot(x ,w))
    first = np.multiply(-y , np.log(predict))
    second = np.multiply((1 -  y) , np.log(1 - predict))
    return np.sum(first - second) /  n
loss =  CostFunction(x ,y ,w )

# Gradient Descent
def GradientDescnet(x ,y , w, alpha , iters):
    cost =  np.zeros(iters)
    m = len(y)
    for i in range(iters):
        predict = sigmoid(np.dot(x , w))
        dw = (1 / m) * np.dot(x.T, (predict - y))
        w = w -  alpha * dw
        cost[i] =  CostFunction(x ,y ,w)
    return cost , w


alpha = 0.1
iters = 2000
new_cost , new_w = GradientDescnet(x ,y ,w ,alpha , iters)
# printing results
print(f"cost before : {loss:.02f}")
print(f'cost after : {new_cost[-1]:.02f}')
print(f'weights before : {w}')
print(f'weights after : {new_w}')

# predictions
def predict(x, w):
    predictions = sigmoid(np.dot(x, w))
    return [1 if i >= 0.5 else 0 for i in predictions]

predictions = predict(x, new_w)
print(f"predictions : {predictions}")

# accuracy
def accuracy(y, predictions):
    correct = np.sum(y == predictions)
    return correct / len(y) * 100

acc = accuracy(y, predictions)
print(f"Accuracy: {acc:.2f}%")
plt.style.use("fivethirtyeight")
plt.plot(range(iters) , new_cost , color = 'orange' , label='loss')
plt.xlabel("iterations")
plt.ylabel("cost variation")
plt.title("Cost Function")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))


plt.scatter(data["Hours_Studied"], data["Passed"], color='red' , s =  100 , label="Actual Data")

x_range = np.linspace(0, 11, 100)
x_plot = np.c_[np.ones(x_range.shape[0]), x_range]
y_prob = sigmoid(np.dot(x_plot, new_w))
plt.plot(x_range, y_prob, color='y', label="Logistic Regression Curve")

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()