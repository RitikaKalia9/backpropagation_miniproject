import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works in any terminal
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Data: truth tables for each logic gate
# ---------------------------------------------------------------------------
# Every gate shares the same four possible input combinations.
INPUTS = torch.tensor(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]]
)

# Expected outputs for each gate, in the same row order as INPUTS.
GATES = {
    "AND":  torch.tensor([[0.0], [0.0], [0.0], [1.0]]),
    "OR":   torch.tensor([[0.0], [1.0], [1.0], [1.0]]),
    "NAND": torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
    "NOR":  torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
    "XOR":  torch.tensor([[0.0], [1.0], [1.0], [0.0]]),
}


# ---------------------------------------------------------------------------
# 2. Model: a small MLP
# ---------------------------------------------------------------------------
class LogicGateMLP(nn.Module):
    """
    A simple feed-forward network:
        Input layer  : 2 neurons  (the two binary inputs)
        Hidden layer : 4 neurons, ReLU activation
        Output layer : 1 neuron,  Sigmoid activation (probability of '1')

    A hidden layer is required because XOR is not linearly separable;
    a single-layer perceptron cannot learn it, but this MLP can.
    """

    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),   # input -> hidden
            nn.ReLU(),                   # non-linearity
            nn.Linear(hidden_size, 1),   # hidden -> output
            nn.Sigmoid()                 # squash output to (0, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Training routine
# ---------------------------------------------------------------------------
def train_model(gate_name: str, epochs: int = 2000, lr: float = 0.05, verbose: bool = True):
    """
    Trains a fresh MLP on the truth table of the given gate.

    Returns (trained_model, final_loss, loss_history) where loss_history is
    a list containing the loss value recorded at every epoch (used later to
    plot training curves).
    """
    torch.manual_seed(42)  # reproducibility

    model = LogicGateMLP()
    criterion = nn.BCELoss()                       # binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    targets = GATES[gate_name]
    loss_history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(INPUTS)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and epoch % 500 == 0:
            print(f"  Epoch {epoch:5d}/{epochs}  Loss: {loss.item():.6f}")

    final_loss = loss.item()
    print(f"Training complete for {gate_name}. Final loss: {final_loss:.6f}")
    return model, final_loss, loss_history


# ---------------------------------------------------------------------------
# 3b. Plotting: training loss curves (used by "Compare all gates")
# ---------------------------------------------------------------------------
PLOTS_DIR = "plots"


def plot_loss_curves(loss_histories: dict, filename: str = "loss_comparison.png"):
    """
    Plots the training loss curve (loss vs. epoch) for every gate on a
    single chart so convergence speed can be compared visually, then
    saves it as a PNG image.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)

    plt.figure(figsize=(8, 5))
    for gate_name, history in loss_histories.items():
        plt.plot(history, label=gate_name)

    plt.title("Training Loss vs. Epoch — All Logic Gates")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Binary Cross-Entropy)")
    plt.yscale("log")  # log scale makes it easier to compare small losses
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"\nLoss comparison plot saved to '{path}'")



def predict(model: nn.Module, a: int, b: int):
    """
    Runs a forward pass for a single (a, b) input pair.
    Returns (predicted_class, confidence_score).
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[float(a), float(b)]])
        raw_output = model(x).item()          # value in (0, 1)
        predicted_class = 1 if raw_output >= 0.5 else 0
        # Confidence = how close the output is to the predicted class
        confidence = raw_output if predicted_class == 1 else (1 - raw_output)
    return predicted_class, confidence


# ---------------------------------------------------------------------------
# 5. Save / load model (optional deliverable)
# ---------------------------------------------------------------------------
SAVE_DIR = "saved_models"


def save_model(model: nn.Module, gate_name: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{gate_name}_model.pth")
    torch.save(model.state_dict(), path)
    print(f"Model saved to '{path}'")


def load_model(gate_name: str):
    path = os.path.join(SAVE_DIR, f"{gate_name}_model.pth")
    if not os.path.exists(path):
        return None
    model = LogicGateMLP()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 6. Helper: get a valid binary input from the user
# ---------------------------------------------------------------------------
def get_binary_input(prompt: str) -> int:
    while True:
        value = input(prompt).strip()
        if value in ("0", "1"):
            return int(value)
        print("Invalid input. Please enter 0 or 1.")


# ---------------------------------------------------------------------------
# 7. Menu-driven main program
# ---------------------------------------------------------------------------
def gate_menu():
    print("\nAvailable Logic Gates:")
    gate_names = list(GATES.keys())
    for i, name in enumerate(gate_names, start=1):
        print(f"  {i}. {name}")
    print(f"  {len(gate_names) + 1}. Back to main menu")

    choice = input("Select a gate by number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(gate_names) + 1):
        print("Invalid selection.")
        return None

    choice = int(choice)
    if choice == len(gate_names) + 1:
        return None
    return gate_names[choice - 1]


def main():
    trained_models = {}  # cache: gate_name -> trained model (per session)

    print("=" * 55)
    print("   LOGIC GATE NEURAL NETWORK PREDICTOR (PyTorch MLP)")
    print("=" * 55)

    while True:
        print("\n----------------- MAIN MENU -----------------")
        print("1. Train a logic gate model")
        print("2. Predict output using a trained model")
        print("3. Load a previously saved model")
        print("4. Compare training performance of all gates")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()

        # ---- Option 1: Train ----
        if choice == "1":
            gate = gate_menu()
            if gate is None:
                continue
            print(f"\nTraining MLP for {gate} gate...")
            model, _, _ = train_model(gate)
            trained_models[gate] = model
            save_choice = input("Save this trained model to disk? (y/n): ").strip().lower()
            if save_choice == "y":
                save_model(model, gate)

        # ---- Option 2: Predict ----
        elif choice == "2":
            if not trained_models:
                print("No trained models in this session yet. Train one first (Option 1) "
                      "or load a saved one (Option 3).")
                continue
            print("\nGates trained this session:", ", ".join(trained_models.keys()))
            gate = input("Enter gate name to use for prediction: ").strip().upper()
            if gate not in trained_models:
                print("That gate has not been trained yet in this session.")
                continue

            # Loop so the user can quickly test multiple input combinations
            # (e.g. all 4 rows of the truth table) without returning to the
            # main menu each time.
            while True:
                a = get_binary_input("Enter first binary input (0 or 1): ")
                b = get_binary_input("Enter second binary input (0 or 1): ")
                pred, conf = predict(trained_models[gate], a, b)
                print(f"\n{gate}({a}, {b}) => Predicted Output: {pred}  "
                      f"(Confidence: {conf * 100:.2f}%)")

                again = input("\nDo you want to continue testing this gate? (y/n): ").strip().lower()
                if again != "y":
                    break

        # ---- Option 3: Load saved model ----
        elif choice == "3":
            gate = input("Enter gate name to load (AND/OR/NAND/NOR/XOR): ").strip().upper()
            if gate not in GATES:
                print("Unknown gate name.")
                continue
            model = load_model(gate)
            if model is None:
                print(f"No saved model found for {gate}. Train and save it first.")
            else:
                trained_models[gate] = model
                print(f"Loaded saved model for {gate}. You can now use Option 2 to predict.")

        # ---- Option 4: Compare all gates ----
        elif choice == "4":
            print("\nTraining all gates for comparison (this may take a few seconds)...\n")
            results = {}
            loss_histories = {}
            for gate_name in GATES:
                print(f"--- {gate_name} ---")
                model, final_loss, history = train_model(gate_name, verbose=False)
                trained_models[gate_name] = model
                results[gate_name] = final_loss
                loss_histories[gate_name] = history

            print("\n===== Final Loss Comparison =====")
            for gate_name, loss_val in results.items():
                print(f"  {gate_name:5s} : final loss = {loss_val:.6f}")
            print("\nObservation: AND, OR, NAND, NOR are linearly separable and "
                  "typically converge to a very low loss quickly. XOR is not "
                  "linearly separable, so it relies on the hidden layer and may "
                  "show a slightly higher or slower-converging loss.")

            # Plot all five training curves on one chart and save as PNG
            plot_loss_curves(loss_histories)

        # ---- Option 5: Exit ----
        elif choice == "5":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()