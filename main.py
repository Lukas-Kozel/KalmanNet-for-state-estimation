import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from KalmanNet import KalmanNet


# Definice třídy pro náš lineární systém
class LinearSystem:
    def __init__(self,Ex0,P0, F, H, Q, R):
        self.Ex0 = Ex0  # očekávaná hodnota počátečního stavu
        self.P0 = P0  # Počáteční kovarianční matice
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.state_dim = F.shape[0]
        self.obs_dim = H.shape[0]

    def get_initial_state(self):
        """
        Vrátí počáteční stav systému.
        """
        return torch.randn(self.state_dim, 1) * torch.sqrt(self.P0) + self.Ex0
    

    def get_fixed_initial_state(self):
        """
        Vrátí počáteční stav systému s fixní hodnotou.
        """
        return self.Ex0.unsqueeze(1)

    def step(self, x_prev):
        w = torch.randn(self.state_dim, 1) * torch.sqrt(self.Q)
        return self.F @ x_prev + w

    def measure(self, x):
        v = torch.randn(self.obs_dim, 1) * torch.sqrt(self.R)
        return self.H @ x + v

# Funkce pro generování dat
def generate_data(system, num_trajectories, seq_len):
    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim)
    for i in range(num_trajectories):
        x = system.get_initial_state()
        # x = system.get_fixed_initial_state()  # Použijeme fixní počáteční stav
        for t in range(seq_len):
            if t>0:
                x = system.step(x)            
            
            y = system.measure(x)
            x_data[i, t, :] = x.squeeze()
            y_data[i, t, :] = y.squeeze()
    return x_data, y_data

# Trénovací funkce
def train(model, train_loader, epochs=50, lr=1e-4, clip_grad=1.0):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Zahajuji trénování KalmanNetu...")
    for epoch in range(epochs):
        for x_true_batch, y_meas_batch in train_loader:
            optimizer.zero_grad()
            x_hat_batch = model(y_meas_batch)
            loss = criterion(x_hat_batch, x_true_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # Ořezání gradientů

            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epocha [{epoch+1}/{epochs}], Chyba (Loss): {loss.item():.6f}')
    print("Trénování dokončeno.")

# Hlavní skript
if __name__ == '__main__':
    # --- 1. Nastavení systémů ---
    F_true = torch.tensor([[0.9]])
    H_true = torch.tensor([[1.0]])
    Q_true = torch.tensor([[0.01]])
    R_true = torch.tensor([[0.01]])
    Ex0_true = torch.tensor([[1.0]])  # Očekávaná hodnota počátečního stavu
    P0_true = torch.tensor([[2.0]])  # Počáteční kov

    sys_true = LinearSystem(Ex0_true,P0_true,F_true, H_true, Q_true, R_true)
    
    # Náš nepřesný model (model mismatch)
    F_model = torch.tensor([[0.8]])  # Horší odhad dynamiky
    H_model = torch.tensor([[1.0]])
    
    #net debug
    # F_model = F_true 
    # H_model = H_true


    Q_model = torch.tensor([[0.0]])
    R_model = torch.tensor([[0.0]])
    Ex0_model = torch.tensor([[0.0]])  # Očekávaná hodnota počátečního stavu
    P0_model = torch.tensor([[0.0]])  # Počáteční kov

    sys_model = LinearSystem(Ex0_model,P0_model,F_model,H_model,Q_model,R_model) # Q a R neznáme
    
    # --- 2. Generování dat ---
    x_train, y_train = generate_data(sys_true, num_trajectories=1000, seq_len=100)
    x_test, y_test = generate_data(sys_true, num_trajectories=1, seq_len=200) # Jedna delší trajektorie pro test
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # --- 3. Trénování modelu ---
    knet = KalmanNet(sys_model)
    train(knet, train_loader)
    
    # --- 4. Evaluace a vizualizace ---
    knet.eval()
    with torch.no_grad():
        x_hat_knet_test = knet(y_test)
        
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