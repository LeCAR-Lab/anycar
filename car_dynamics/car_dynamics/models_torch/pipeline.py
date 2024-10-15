import torch
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import track

def train_MLP_pipeline(logger, model, train_loader, test_loader, device, save_dir, num_epochs=10000, batch_size=4096, lr=0.001):
    """Train a MLP model with the given train_loader and save the model to save_dir.
    Args:
        model: MLP model
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        device: torch.device
        save_dir: path to save the model
        num_epochs: number of epochs to train
        batch_size: batch size
        lr: learning rate
    Returns:
        dict: {'test_loss': test_loss_list, 'train_loss': train_loss_list}
    """
    train_loss_list = []
    test_loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    logger.watch(model, 'all', log_freq=2)
    for epoch in track(range(num_epochs)):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # outputs: (B, 2)-> 2dim: beta, delta, inputs (B, 40)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            

        model.eval()
        with torch.no_grad():
            test_loss = 0.
            if test_loader is not None:
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model.predict(inputs)
                    # test_loss += criterion(outputs, targets).item()
                    test_loss = torch.mean((outputs - targets) ** 2, dim=0) + test_loss
                
                test_loss /= len(test_loader)
            
            train_loss = 0.
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                train_loss += criterion(outputs, targets).item()
            train_loss /= len(train_loader)
            
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        log_dict = {"train_loss": train_loss}
        log_dict.update({f"test_loss_{i}": test_loss[i] for i in range(test_loss.shape[0])})
        logger.log( log_dict )
        # print(f"Epoch {epoch}, Validation Loss: {valid_loss}")
    print(model.spec)
    model.save(save_dir)
    return {'test_loss': test_loss_list, 'train_loss': train_loss_list}


def train_MLP_kbm_pipeline(logger, model, train_loader, test_loader, device, save_dir, dt, num_epochs=10000, batch_size=4096, lr=0.001):
    """Train a MLP model with the given train_loader and save the model to save_dir.
    Args:
        model: MLP model
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        device: torch.device
        save_dir: path to save the model
        num_epochs: number of epochs to train
        batch_size: batch size
        lr: learning rate
    Returns:
        dict: {'test_loss': test_loss_list, 'train_loss': train_loss_list}
    """
    train_loss_list = []
    test_loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def pred_loss(inputs, outputs, targets):
        # inputs [x, y, cos(psi), sin(psi), v, throttle, steer]
        # outputs [LF, LR, delta_proj, delta_shift, vel_shift, vel_proj]
        # tragets [dx, dy, cos(dpsi), sin(dpsi), dv]
        batch_size = inputs.shape[0]
        inputs_unfold = inputs.view(batch_size, -1, 7)
        x = inputs_unfold[:, -1, 0]
        y = inputs_unfold[:, -1, 1]
        psi = torch.atan2(inputs_unfold[:, -1, 3], inputs_unfold[:, -1, 2])
        v = inputs_unfold[:, -1, 4]
        throttle = inputs_unfold[:, -1, 5]
        steer = inputs_unfold[:, -1, 6]
        
        
        LF = .16
        LR = .15
        
        delta_proj = torch.sigmoid( outputs[:, 0] ) - 0.9
        delta_shift = torch.sigmoid ( outputs[:, 1] ) * 2. - 1.
        # LF = torch.tanh(outputs[:, 0]) * 0.05 + 0.15
        # LR = torch.tanh(outputs[:, 1]) * 0.05 + 0.15
        # delta_proj = torch.tanh(outputs[:, 2])
        # delta_shift = torch.tanh(outputs[:, 3]) * torch.pi
        
        # LF = torch.clip(LF, 0.01, torch.inf)
        # LR = torch.clip(LR, 0.01, torch.inf)
        # vel_proj = outputs[:, 4]
        # vel_shift = outputs[:, 5]
        # prev_vel_proj = outputs[:, 6]
        
        # vel_proj = outputs[:, 2]
        # vel_shift = outputs[:, 3]
        # prev_vel_proj = outputs[:, 4]
        
        beta = torch.atan(torch.tan(delta_proj * steer + delta_shift) * LF / (LF + LR))
        pred_next = torch.zeros_like(targets).to(device)
        pred_next[:, 0] = v * torch.cos(psi + beta)
        pred_next[:, 1] = v * torch.sin(psi + beta)
        pred_next[:, 2] = torch.cos(v / LR * torch.sin(beta) * dt)
        pred_next[:, 3] = torch.sin(v / LR * torch.sin(beta) * dt)
        # print(pred_next.shape, targets.shape)
        ## Temporarily remove dvel
        # pred_next[:, 4] = vel_proj * throttle + prev_vel_proj * v + vel_shift
        # pred_next[:, 4] = vel_proj * throttle + vel_shift
        
        ## Temporary only keep dvel
        # pred_next[:, 0] = vel_proj * throttle + prev_vel_proj * v + vel_shift
        
        # negative_penalty_dim0 = torch.norm(torch.clip(-outputs[:, 0], 0, torch.inf)).mean() * 0.01
        # negative_penalty_dim1 = torch.norm(torch.clip(-outputs[:, 1], 0, torch.inf)).mean() * 0.01
        # return torch.nn.MSELoss()(pred_next, targets) + negative_penalty_dim1 + negative_penalty_dim0
        ## Add L2 penalty to delta_shift to make it close to 0
        ## Change to L1 penalty
        # loss_shift = torch.norm(delta_shift, p=1) * 0.0001
        loss_shift = torch.norm(delta_shift) * 0.0
        loss_proj = torch.norm(delta_proj) * 0.0
        return torch.nn.MSELoss()(pred_next, targets)  + loss_shift + loss_proj

    for epoch in track(range(num_epochs)):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            # outputs = model(inputs) # outputs: (B, 2)-> 2dim: beta, delta, inputs (B, 40)
            outputs = model(inputs[:, :-2])
            loss = pred_loss(inputs, outputs, targets)
            loss.backward()
            optimizer.step()
            

        model.eval()
        with torch.no_grad():
            test_loss = 0.
            if test_loader is not None:
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    # outputs = model(inputs)
                    outputs = model(inputs[:, :-2])
                    test_loss += pred_loss(inputs, outputs, targets).item()
                
                test_loss /= len(test_loader)
            
            train_loss = 0.
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                # outputs = model(inputs)
                outputs = model(inputs[:, :-2])
                train_loss += pred_loss(inputs, outputs, targets).item()
            train_loss /= len(train_loader)
            
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        logger.log({"train_loss": train_loss, "test_loss": test_loss})
        # print(f"Epoch {epoch}, Validation Loss: {valid_loss}")
        
    model.save(save_dir)
    return {'test_loss': test_loss_list, 'train_loss': train_loss_list}



def train_mlp_dbm_pipeline(logger, model, dynamics, 
                           train_loader, test_loader, batch_size,
                           device, save_dir, num_epochs=10000, lr=0.001):
    """Train a MLP model with embeded Dynamic Bicycle Model (dbm) with the given train_loader and save the model to save_dir.
    Args:
        model: MLP model
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        device: torch.device
        save_dir: path to save the model
        num_epochs: number of epochs to train
        batch_size: batch size
        lr: learning rate
    Returns:
        dict: {'test_loss': test_loss_list, 'train_loss': train_loss_list}
    """
    train_loss_list = []
    test_loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def pred_loss(inputs, outputs, targets):
        # inputs [x, y, theta, vx, vy, omega, action_0, action_1]
        # outputs [steer_proj, steer_shift, mass, I_z, B_f, C_f, D_f, B_r, C_r, D_r]
        # tragets [dx, dy, dtheta, dvx, dvy, domega]
        batch_size = inputs.shape[0]
        inputs_unfold = inputs.view(batch_size, -1, 8)
        x = inputs_unfold[:, -1, 0]
        y = inputs_unfold[:, -1, 1]
        theta = inputs_unfold[:, -1, 2]
        vx = inputs_unfold[:, -1, 3]
        vy = inputs_unfold[:, -1, 4]
        omega = inputs_unfold[:, -1, 5]
        action_0 = inputs_unfold[:, -1, 6]
        action_1 = inputs_unfold[:, -1, 7]
        
        K_RFY = torch.sigmoid(outputs[:, 0]) * 300.
        K_FFY = torch.sigmoid(outputs[:, 1]) * 300.
        Sa = torch.sigmoid(outputs[:, 2])
        Sb = torch.sigmoid(outputs[:, 3]) * 2. - 1.
        
        # print(K_RFY.mean(dim=0), K_FFY.mean(dim=0), Sa.mean(dim=0), Sb.mean(dim=0))
    
        dynamics.batch_Sa = Sa
        dynamics.batch_Sb = Sb
        dynamics.batch_K_RFY = K_RFY
        dynamics.batch_K_FFY = K_FFY
        
        next_x, next_y, next_psi, next_vx, next_vy, next_omega = dynamics.step(
                                                x, y, theta, vx, vy, omega, action_0, action_1)

        pred_next = torch.stack([next_x, next_y, next_psi, next_vx, next_vy, next_omega], dim=1)
        
        return torch.nn.MSELoss()(pred_next[:, 3:], targets[:, 3:]) , {'K_RFY': K_RFY.mean(), 'K_FFY': K_FFY.mean(), 'Sa': Sa.mean(), 'Sb': Sb.mean()}

    logger.watch(model, log_freq=10)
    for epoch in track(range(num_epochs)):
        model.train()
        for inputs, targets in train_loader:
            if inputs.shape[0] != batch_size:
                continue
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # outputs: (B, 2)-> 2dim: beta, delta, inputs (B, 40)
            # outputs = model(inputs[:, :-2])
            loss, _ = pred_loss(inputs, outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = 0.
            if test_loader is not None:
                for inputs, targets in test_loader:
                    if inputs.shape[0] != batch_size:
                        continue
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    # outputs = model(inputs[:, :-2])
                    loss, _ = pred_loss(inputs, outputs, targets)
                    test_loss += loss.item()
                
                test_loss /= len(test_loader)
            
            train_loss = 0.
            for inputs, targets in train_loader:
                if inputs.shape[0] != batch_size:
                    continue
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                # outputs = model(inputs[:, :-2])
                loss, info = pred_loss(inputs, outputs, targets)
                train_loss += loss.item()
                train_loss /= len(train_loader)
                logger.log(info)
            
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        logger.log({"train_loss": train_loss, "test_loss": test_loss})
        # print(f"Epoch {epoch}, Validation Loss: {valid_loss}")
        
    model.save(save_dir)
    return {'test_loss': test_loss_list, 'train_loss': train_loss_list}


def train_MLP_kbm_pipeline(logger, model, train_loader, test_loader, device, save_dir, dt, num_epochs=10000, batch_size=4096, lr=0.001):
    """Train a MLP model with the given train_loader and save the model to save_dir.
    Args:
        model: MLP model
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        device: torch.device
        save_dir: path to save the model
        num_epochs: number of epochs to train
        batch_size: batch size
        lr: learning rate
    Returns:
        dict: {'test_loss': test_loss_list, 'train_loss': train_loss_list}
    """
    train_loss_list = []
    test_loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def pred_loss(inputs, outputs, targets):
        # inputs [x, y, cos(psi), sin(psi), v, throttle, steer]
        # outputs [LF, LR, delta_proj, delta_shift, vel_shift, vel_proj]
        # tragets [dx, dy, cos(dpsi), sin(dpsi), dv]
        batch_size = inputs.shape[0]
        inputs_unfold = inputs.view(batch_size, -1, 7)
        x = inputs_unfold[:, -1, 0]
        y = inputs_unfold[:, -1, 1]
        psi = torch.atan2(inputs_unfold[:, -1, 3], inputs_unfold[:, -1, 2])
        v = inputs_unfold[:, -1, 4]
        throttle = inputs_unfold[:, -1, 5]
        steer = inputs_unfold[:, -1, 6]
        LF = outputs[:, 0]
        LR = outputs[:, 1]
        delta_proj = outputs[:, 2]
        delta_shift = outputs[:, 3]
        vel_proj = outputs[:, 4]
        vel_shift = outputs[:, 5]
        # prev_vel_proj = outputs[:, 6]
        beta = torch.atan(torch.tan(delta_proj * steer + delta_shift) * LF / (LF + LR))
        pred_next = torch.zeros_like(targets).to(device)
        pred_next[:, 0] = v * torch.cos(psi + beta)
        pred_next[:, 1] = v * torch.sin(psi + beta)
        pred_next[:, 2] = torch.cos(v / LR * torch.sin(beta))
        pred_next[:, 3] = torch.sin(v / LR * torch.sin(beta))
        # pred_next[:, 4] = vel_proj * throttle + prev_vel_proj * v + vel_shift
        pred_next[:, 4] = vel_proj * throttle + vel_shift
        
        # negative_penalty_dim0 = torch.norm(torch.clip(-outputs[:, 0], 0, torch.inf)).mean() * 0.01
        # negative_penalty_dim1 = torch.norm(torch.clip(-outputs[:, 1], 0, torch.inf)).mean() * 0.01
        # return torch.nn.MSELoss()(pred_next, targets) + negative_penalty_dim1 + negative_penalty_dim0
        return torch.nn.MSELoss()(pred_next, targets) 

    for epoch in track(range(num_epochs)):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # outputs: (B, 2)-> 2dim: beta, delta, inputs (B, 40)
            loss = pred_loss(inputs, outputs, targets)
            loss.backward()
            optimizer.step()
            

        model.eval()
        with torch.no_grad():
            test_loss = 0.
            if test_loader is not None:
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    test_loss += pred_loss(inputs, outputs, targets).item()
                
                test_loss /= len(test_loader)
            
            train_loss = 0.
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                train_loss += pred_loss(inputs, outputs, targets).item()
            train_loss /= len(train_loader)
            
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)
        logger.log({"train_loss": train_loss, "test_loss": test_loss})
        # print(f"Epoch {epoch}, Validation Loss: {valid_loss}")
        
    model.save(save_dir)
    return {'test_loss': test_loss_list, 'train_loss': train_loss_list}

