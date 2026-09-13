from autograd.nn import MLP

# Initialize model
model = MLP(4, [1])

# Data
xs = [
    [1, 6, 4, 2],
    [1, 5, 7, 6],
    [1, 4, 4, 2],
    [1, 2, 7, 3],
    [1, 1, 8, 1],
    [1, 8, 3, 3],
    [1, 9, 2, 6]
]
y = [6, 5, 4, 2, 1, 8, 9]

learning_rate = 0.01
epochs = 500

for k in range(epochs):
    
    # 1. Forward Pass
    ypred = [model(x) for x in xs]
    
    # 2. Compute MSE Loss
    loss = sum((ys - yp)**2 for ys, yp in zip(y, ypred)) * (1.0 / len(y))
    
    # 3. Zero Gradients
    model.zero_grad()
    
    # 4. Backward Pass
    loss.backward()
    
    # 5. Gradient Descent (Update Parameters)
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        
    if k % 50 == 0:
        print(f"Epoch {k:3d} | Loss: {loss.data:.4f}")

print(f"\nFinal Loss: {loss.data:.4f}")

print("\n--- Model Predictions vs Targets ---")
for x_input, target in zip(xs, y):
    prediction = model(x_input)
    print(f"Input: {x_input} -> Predicted: {prediction.data:.2f} | Target: {target}")

# Test on a completely new test input:
test_sample = [1, 6, 7, 2]
test_pred = model(test_sample)
print(f"\nNew Sample Prediction: {test_sample} -> {test_pred.data:.2f}")

