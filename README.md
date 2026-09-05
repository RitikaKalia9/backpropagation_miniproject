# Logic Gate Neural Network Predictor

A mini-project demonstrating **forward propagation, backpropagation, binary cross-entropy loss, and Adam optimization** using a small PyTorch Multi-Layer Perceptron (MLP).

The project trains a neural network to learn the truth tables of five basic logic gates:

- AND
- OR
- NAND
- NOR
- XOR

The project also supports prediction, saving/loading trained models, and comparing the training performance of all gates.

## Features

- Train an individual logic gate neural network
- Predict the output for custom binary inputs
- Save trained models as `.pth` files
- Load previously saved models
- Train and compare all five gates
- Generate a training-loss comparison plot
- Demonstrates backpropagation using PyTorch autograd

## Model Architecture

The network is a small feed-forward MLP:

```text
Input Layer (2 neurons)
        ↓
Linear Layer
        ↓
Hidden Layer (4 neurons)
        ↓
ReLU
        ↓
Linear Layer
        ↓
Output Layer (1 neuron)
        ↓
Sigmoid
```

### Why a hidden layer?

AND, OR, NAND, and NOR are linearly separable and can be learned by a single-layer perceptron.

XOR is **not linearly separable**, so a hidden layer with a non-linear activation function is required for the network to learn it.

## Training

For each gate, the model is trained using:

- **Loss function:** Binary Cross-Entropy (`BCELoss`)
- **Optimizer:** Adam
- **Learning rate:** `0.05`
- **Epochs:** `2000`
- **Hidden neurons:** `4`
- **Random seed:** `42`

During each training iteration:

1. The inputs are passed through the network.
2. The predicted outputs are compared with the target truth table.
3. Binary cross-entropy loss is calculated.
4. `loss.backward()` performs backpropagation and computes gradients.
5. `optimizer.step()` updates the model parameters.

## Logic Gate Truth Tables

| A | B | AND | OR | NAND | NOR | XOR |
|---|---|-----|----|------|-----|-----|
| 0 | 0 |  0  | 0  |  1   |  1  | 0 |
| 0 | 1 |  0  | 1  |  1   |  0  | 1 |
| 1 | 0 |  0  | 1  |  1   |  0  | 1 |
| 1 | 1 |  1  | 1  |  0   |  0  | 0 |

## Project Structure

```text
backpropagation_miniproject-master/
│
├── app.py
├── requirements.txt
├── plots/
│   └── loss_comparison.png
├── saved_models/          # Created when models are saved
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Matplotlib 3.5+

## Installation

Clone or download the project and open a terminal in the project directory.

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python app.py
```

You will see the main menu:

```text
1. Train a logic gate model
2. Predict output using a trained model
3. Load a previously saved model
4. Compare training performance of all gates
5. Exit
```

## How to Use

### 1. Train a Logic Gate

Select option `1`, choose a gate, and the model will be trained from scratch.

For example:

```text
Training MLP for XOR gate...
```

After training, you can optionally save the model.

### 2. Make Predictions

Select option `2` and choose one of the gates trained during the current session.

Enter two binary inputs:

```text
Enter first binary input (0 or 1): 1
Enter second binary input (0 or 1): 0

XOR(1, 0) => Predicted Output: 1
```

The program also displays the model's confidence.

### 3. Save and Load Models

After training, choose `y` when asked whether to save the model.

Models are stored in:

```text
saved_models/
```

For example:

```text
saved_models/XOR_model.pth
```

To load a saved model, select option `3` and enter the gate name.

### 4. Compare All Gates

Select option `4` to train all five gates and compare their final losses.

The program also generates:

```text
plots/loss_comparison.png
```

This plot shows the loss-versus-epoch curves for all five logic gates.

## Backpropagation Concept

This project demonstrates the basic backpropagation workflow:

```text
Input
  ↓
Forward Propagation
  ↓
Predicted Output
  ↓
Calculate Loss
  ↓
Backpropagation
  ↓
Calculate Gradients
  ↓
Update Weights
  ↓
Repeat
```

In PyTorch, the key line is:

```python
loss.backward()
```

This uses automatic differentiation to calculate the gradients of the loss with respect to the network parameters.

The optimizer then updates the parameters:

```python
optimizer.step()
```

Before calculating gradients for the next iteration, existing gradients are cleared using:

```python
optimizer.zero_grad()
```

## Technologies Used

- **Python**
- **PyTorch**
- **Matplotlib**

## Learning Objectives

This mini-project is intended to provide practical understanding of:

- Neural network architecture
- Forward propagation
- Activation functions
- ReLU
- Sigmoid
- Binary cross-entropy loss
- Backpropagation
- Gradients
- Adam optimizer
- Model training
- Model saving and loading
- Linear vs. non-linear separability
- Training-loss visualization

## Limitations

This is an educational project using only four possible input combinations for each logic gate. It is designed to demonstrate neural-network concepts rather than solve a real-world classification problem.

The confidence shown by the program is based on the sigmoid output and should be interpreted as the model's prediction confidence, not as a calibrated probability.

## License

This project is intended for educational and academic use.
