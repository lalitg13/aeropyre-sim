import numpy as np

class AutonomousUAV:
    def __init__(self, uav_id, init_x, init_y, max_speed=4.5, payload=1.0):
        self.id = uav_id
        self.x = float(init_x)
        self.y = float(init_y)
        self.max_speed = max_speed
        self.payload = payload
        self.trajectory = [(self.x, self.y)]
        self.drops = []
        
    def step_toward(self, target_x, target_y, env):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 1e-3:
            step_size = min(self.max_speed, dist)
            self.x += (dx / dist) * step_size
            self.y += (dy / dist) * step_size
            
        self.trajectory.append((self.x, self.y))
        
        # Check if UAV is over critical fire boundary (> 550 K) and drop retardant
        grid_i = int(np.clip(self.y / env.dx, 0, env.N - 1))
        grid_j = int(np.clip(self.x / env.dx, 0, env.N - 1))
        
        if env.temperature[grid_i, grid_j] > 480.0 and self.payload > 0.05:
            # Drop retardant in 3x3 radius
            drop_amount = 0.25
            self.payload -= 0.05
            self.drops.append((self.x, self.y))
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = grid_i + di, grid_j + dj
                    if 0 <= ni < env.N and 0 <= nj < env.N:
                        env.retardant[ni, nj] = min(1.0, env.retardant[ni, nj] + drop_amount)

class DroneSwarm:
    def __init__(self, num_drones=12, env_width=600.0):
        self.drones = []
        for i in range(num_drones):
            # Deploy drones along southern perimeter
            init_x = 50.0 + i * (env_width - 100.0) / (num_drones - 1)
            init_y = 30.0
            self.drones.append(AutonomousUAV(i, init_x, init_y))
            
    def update(self, env):
        # Identify active fire perimeter coordinates
        active_mask = (env.temperature > 520.0) & (env.temperature < 850.0)
        perimeter_indices = np.argwhere(active_mask)
        
        if len(perimeter_indices) == 0:
            targets = [((env.N/2)*env.dx, (env.N/2)*env.dx) for _ in self.drones]
        else:
            # Sort perimeter by danger (downwind front)
            front_scores = perimeter_indices[:, 0] * 1.5 + perimeter_indices[:, 1] * 1.2
            sorted_idx = np.argsort(-front_scores)
            selected_indices = perimeter_indices[sorted_idx]
            
            targets = []
            for d in self.drones:
                # Assign closest available perimeter segment
                target_cand = selected_indices[d.id % len(selected_indices)]
                target_y = target_cand[0] * env.dx
                target_x = target_cand[1] * env.dx
                targets.append((target_x, target_y))
                
        for drone, (tx, ty) in zip(self.drones, targets):
            drone.step_toward(tx, ty, env)
