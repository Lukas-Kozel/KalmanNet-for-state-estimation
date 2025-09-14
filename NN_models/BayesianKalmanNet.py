import torch
import torch.nn as nn
import torch.nn.functional as F # Správný import pro relu

class BayesianKalmanNet(nn.Module):
    """
    BayesianKalmanNet Architektura #1 pro nelineární systémy.
    Tato verze je plně "device-aware" a přijímá zařízení jako
    parametr v konstruktoru.
    """
    def __init__(self, system_model, device, hidden_size_multiplier=10, dropout_prob=0.2):
        super(BayesianKalmanNet, self).__init__()
        
        # Uložíme si zařízení jako atribut třídy
        self.device = device
        
        self.system_model = system_model
        self.state_dim = system_model.state_dim
        self.obs_dim = system_model.obs_dim

        self.dropout_prob = dropout_prob

        # Heuristika pro velikost skrytého stavu
        moments_dim = self.state_dim*self.state_dim + self.obs_dim*self.obs_dim
        self.hidden_dim = moments_dim * hidden_size_multiplier

        # Vstupní dimenze
        input_dim = self.state_dim + self.obs_dim

        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.state_dim * self.obs_dim)

        self.dropout2 = nn.Dropout(self.dropout_prob)

        # vmap pro nelineární funkce
        self.f_vmap = torch.vmap(self.system_model.f, in_dims=(0,))
        self.h_vmap = torch.vmap(self.system_model.h, in_dims=(0,))

    def forward(self, y_seq, num_samples=20):
        """
        Tato metoda nyní provádí POUZE JEDEN dopředný průchod pro danou dávku.
        Je ideální pro trénink.
        """
        batch_size, seq_len, _ = y_seq.shape
        
        # Toto `x̂_{t-1|t-1}` se bude aktualizovat v každém kroku smyčky.
        x_hat_filtered_prev = torch.zeros(batch_size, self.state_dim, device=self.device)

        # Inicializace změnu v odhadu stavu. Toto je vstup pro neuronovou síť.
        # Na začátku je změna nulová.
        delta_x_prev = torch.zeros(batch_size, self.state_dim, device=self.device)

        # Inicializace skrytý stav `h` pro GRU. Každá z `num_samples` (J)
        # realizací musí mít svůj vlastní, nezávislý skrytý stav, aby si
        # pamatovala vlastní historii.
        # Rozměry: [J, 1, batch_size, hidden_dim]
        h_gru_ensemble = torch.zeros(num_samples,1, batch_size, self.hidden_dim, device=self.device)

        # Seznamy pro sběr finálních výsledků (průměr a kovariance) pro každý časový krok.
        x_filtered_trajectory = []
        P_filtered_trajectory = []

        # Zpracování dat krok po kroku, od t=0 do t=seq_len-1.
        for t in range(seq_len):
            y_t = y_seq[:, t, :] # Aktuální měření pro všechny trajektorie v dávce
            
            # Seznam pro uložení J různých odhadů stavu v tomto jednom kroku `t`.
            x_hat_posterior_ensemble_t = []

            # Seznam pro uložení nových skrytých stavů GRU.
            new_h_gru_list =[]

            # --- SMYČKA PŘES J REALIZACÍ UVNITŘ ČASOVÉ SMYČKY ---
            # `num_samples` (J) dopředných průchodů, pro získání
            # ensemble odhadů. Každý průchod je jiný díky dropoutu.
            for j in range(num_samples):

                # --- Kalmanovská predikce ---
                # Všechny J realizace vychází ze STEJNÉHO průměrného odhadu z minulého kroku.
                # rovnice `x̂_{t|t-1} = f(x̂_{t-1|t-1})`
                x_hat_predicted = self.f_vmap(x_hat_filtered_prev)

                # rovnice `ŷ_{t|t-1} = h(x̂_{t|t-1})`
                y_hat = self.h_vmap(x_hat_predicted)

                # Inovace (rozdíl mezi skutečným a predikovaným měřením)
                innovation = y_t - y_hat

                # --- Příprava vstupu pro neuronovou síť ---
                # Normalizace vstupů pro lepší stabilitu tréninku
                norm_innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
                norm_delta_x = F.normalize(delta_x_prev, p=2, dim=1, eps=1e-12)
                nn_input = torch.cat([norm_delta_x, norm_innovation], dim=1)


                # --- Průchod neuronovou sítí ---
                activated_input = F.relu(self.input_layer(nn_input))
                activated_input_dropped = self.dropout1(activated_input)

                h_j_old = h_gru_ensemble[j]
                # GRU očekává vstup ve tvaru `[seq_len, batch_size, input_dim]`.
                out_gru, h_j_new = self.gru(activated_input_dropped.unsqueeze(0), h_j_old) # unsqueeze pro úpravu dimenze

                new_h_gru_list.append(h_j_new)

                out_gru_squeezed = out_gru.squeeze(0) # squeeze pro odstranění upravené dimenze

                # --- Výpočet stochastického Kalmanova zisku ---
                K_vec_raw = self.output_layer(out_gru_squeezed)
                K_vec = self.dropout2(K_vec_raw)
                K = K_vec.reshape(batch_size, self.state_dim, self.obs_dim)


                # --- Kalmanova aktualizace ---
                correction = (K @ innovation.unsqueeze(-1)).squeeze(-1)

                # Toto odpovídá rovnici `x̂_{t|t} = x̂_{t|t-1} + K_t * innovation_t`
                x_hat_filtered_j = x_hat_predicted + correction

                x_hat_posterior_ensemble_t.append(x_hat_filtered_j)

            # Nahrazení starého ensemble skrytých stavů novým.
            h_gru_ensemble = torch.stack(new_h_gru_list, dim=0)


            # --- ZPRŮMĚROVÁNÍ VÝSLEDKŮ PRO KROK t ---
            ensemble_at_t = torch.stack(x_hat_posterior_ensemble_t, dim=0)

            # Rovnice (22a): Průměrný odhad
            x_hat_filtered_at_t = torch.mean(ensemble_at_t, dim=0)

            # Rovnice (22b): Odhad kovariance pro tento krok t
            diff = ensemble_at_t - x_hat_filtered_at_t
            P_filtered_at_t = (diff.unsqueeze(-1)*diff.unsqueeze(-2)).mean(dim=0)

            # Uložení výsledků pro tento časový krok
            x_filtered_trajectory.append(x_hat_filtered_at_t)
            P_filtered_trajectory.append(P_filtered_at_t)

            # --- AKTUALIZACE PRO DALŠÍ KROK t+1 ---
            # ZPRŮMĚROVANÝ odhad se stává vstupem pro další krok
            delta_x_prev = x_hat_filtered_at_t-self.f_vmap(x_hat_filtered_prev)
            x_hat_filtered_prev = x_hat_filtered_at_t.clone()
            
        return torch.stack(x_filtered_trajectory, dim=1), torch.stack(P_filtered_trajectory, dim=1)