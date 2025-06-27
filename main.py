import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from KalmanNet import KalmanNet
from utils import generate_data, train
from LinearSystem import LinearSystem


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používané zařízení: {device}")

    # --- Reálný systém ---
    F_true = torch.tensor([[0.9]])
    H_true = torch.tensor([[1.0]])
    Q_true = torch.tensor([[0.01]])
    R_true = torch.tensor([[0.01]])
    Ex0_true = torch.tensor([[1.0]])  # Očekávaná hodnota počátečního stavu
    P0_true = torch.tensor([[2.0]])  # Počáteční kov

    sys_true = LinearSystem(Ex0_true,P0_true,F_true, H_true, Q_true, R_true)
    
    # nepřesný model (model mismatch)
    F_model = torch.tensor([[0.8]])  # Horší odhad dynamiky
    H_model = torch.tensor([[1.0]])
    Q_model = torch.tensor([[0.0]])
    R_model = torch.tensor([[0.0]])
    Ex0_model = torch.tensor([[0.0]])  # Očekávaná hodnota počátečního stavu
    P0_model = torch.tensor([[0.0]])  # Počáteční kov

    sys_model = LinearSystem(Ex0_model,P0_model,F_model,H_model,Q_model,R_model) # Q a R neznáme
    
    # --- Generování dat ---
    x_train, y_train = generate_data(sys_true, num_trajectories=1000, seq_len=100)
    x_test, y_test = generate_data(sys_true, num_trajectories=1, seq_len=200) # Jedna delší trajektorie pro test
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # --- Trénování modelu ---
    knet = KalmanNet(sys_model).to(device)
    train(knet, train_loader,device)
    
    # --- Evaluace a vizualizace ---
    knet.eval()
    with torch.no_grad():
        y_test_device = y_test.to(device)
        x_hat_knet_test_device = knet(y_test_device)
        x_hat_knet_test = x_hat_knet_test_device.cpu()
        
    mse_knet = nn.MSELoss()(x_hat_knet_test, x_test)
    
    print(f"\nVýsledná MSE na testovací sadě: {mse_knet.item():.4f}")
    
    plt.figure(figsize=(15, 6))
    plt.title("Výsledky KalmanNetu na testovací trajektorii")
    plt.plot(x_test[0].numpy(), 'k-', linewidth=2, label="Skutečný stav (Ground Truth)")
    plt.plot(y_test[0].numpy(), 'r.', markersize=4, label="Měření")
    plt.plot(x_hat_knet_test[0].numpy(), 'g--', linewidth=2, label=f"Odhad KalmanNet (MSE={mse_knet.item():.4f})")
    plt.xlabel("Časový krok")
    plt.ylabel("Hodnota")
    plt.grid(True)
    plt.legend()
    plt.show()