import torch
import torch.nn as nn
from copy import deepcopy

# Funkce pro generování dat
def generate_data(system, num_trajectories, seq_len):

    # Zjištění zařízení, na kterém systém pracuje
    device = system.Ex0.device

    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim,device=device)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim,device=device)
    for i in range(num_trajectories):
        x = system.get_initial_state()
        # x = system.get_deterministic_initial_state()  # Použijeme fixní počáteční stav
        for t in range(seq_len):
            if t>0:
                x = system.step(x)            
            
            y = system.measure(x)
            x_data[i, t, :] = x.squeeze()
            y_data[i, t, :] = y.squeeze()
    return x_data, y_data

# Trénovací funkce
def train(model, train_loader,device, epochs=50, lr=1e-4, clip_grad=1.0):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Zahajuji trénování KalmanNetu...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_norm = 0.0

        for x_true_batch, y_meas_batch in train_loader:

            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()
            x_hat_batch = model(y_meas_batch)
            
            loss = criterion(x_hat_batch, x_true_batch)

            loss.backward()
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            print(f'Epocha [{epoch+1}/{epochs}], Prům. chyba: {avg_loss:.6f}, Celková norma grad.: {total_norm:.4f}')
    print("Trénování dokončeno.")

def train_with_scheduler(model, train_loader, val_loader, device, 
                         epochs=200, lr=1e-3, clip_grad=1.0, 
                         early_stopping_patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_true_batch, y_meas_batch in train_loader:
            x_true_batch = x_true_batch.to(device)
            y_meas_batch = y_meas_batch.to(device)

            optimizer.zero_grad()
            x_hat_batch = model(y_meas_batch)
            loss = criterion(x_hat_batch, x_true_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_true_val, y_meas_val in val_loader:
                x_true_val = x_true_val.to(device)
                y_meas_val = y_meas_val.to(device)
                x_hat_val = model(y_meas_val)
                loss = criterion(x_hat_val, x_true_val)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epocha [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping spuštěno po {epoch + 1} epochách.")
            break
            
    print("Trénování dokončeno.")
    
    if best_model_state:
        print(f"Načítám nejlepší model s validační chybou: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)
        
    return model