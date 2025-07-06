import torch
import torch.nn as nn

# Funkce pro generování dat
def generate_data(system, num_trajectories, seq_len):
    x_data = torch.zeros(num_trajectories, seq_len, system.state_dim)
    y_data = torch.zeros(num_trajectories, seq_len, system.obs_dim)
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