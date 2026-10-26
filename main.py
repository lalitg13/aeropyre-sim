import os
import numpy as np
import matplotlib.pyplot as plt
from src.physics import WildfireEnvironment
from src.pinn_model import PINNSurrogate
from src.swarm import DroneSwarm

def run_simulation():
    print("[*] Initializing AeroPyre-Sim Wildfire Environment...")
    env = WildfireEnvironment(grid_size=60, dx=10.0, dt=0.5)
    swarm = DroneSwarm(num_drones=12, env_width=600.0)
    pinn = PINNSurrogate(input_dim=3, hidden_dim=32, output_dim=1, lr=0.008)
    
    total_steps = 100
    containment_history = []
    initial_fuel = np.sum(env.fuel)
    
    print(f"[*] Simulating {total_steps} timesteps with autonomous swarm & PINN guidance...")
    for t in range(total_steps):
        env.step()
        swarm.update(env)
        
        # Synthetic sampling for PINN collocation and data loss
        sample_indices_i = np.random.randint(0, env.N, 40)
        sample_indices_j = np.random.randint(0, env.N, 40)
        sample_x = sample_indices_j * env.dx / 600.0
        sample_y = sample_indices_i * env.dx / 600.0
        sample_t = np.full_like(sample_x, (t * env.dt) / 50.0)
        
        X_train = np.stack([sample_x, sample_y, sample_t], axis=1)
        y_train = (env.temperature[sample_indices_i, sample_indices_j].reshape(-1, 1) - 295.0) / 700.0
        
        # Collocation points for PDE residual
        X_colloc = np.random.rand(30, 3)
        pinn.train_step(X_train, y_train, X_colloc, env)
        
        # Containment metric: 1 - ratio of burned area vs worst-case unchecked spread
        burned_cells = np.sum(env.fuel < 0.2)
        containment_rate = max(0.0, 100.0 * (1.0 - (burned_cells / (env.N * env.N * 0.45))))
        containment_history.append(containment_rate)
        
        if (t + 1) % 25 == 0:
            print(f"    Step {t+1:03d}/{total_steps} | Max Temp: {np.max(env.temperature):.1f} K | Containment: {containment_rate:.1f}%")
            
    print("[*] Generating high-resolution simulation telemetry...")
    os.makedirs("assets", exist_ok=True)
    
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    
    # Panel A: 2D Thermal Wildfire Heatmap with Elevation Contours
    ax_heat = axes[0, 0]
    im_heat = ax_heat.imshow(
        env.temperature - 273.15,
        extent=[0, 600, 0, 600],
        origin='lower',
        cmap='inferno',
        vmin=20,
        vmax=850
    )
    cbar1 = fig.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar1.set_label("Temperature (°C)", color="#E2E8F0", fontsize=10)
    contours = ax_heat.contour(env.X, env.Y, env.elevation, levels=6, colors="#94A3B8", alpha=0.4, linewidths=0.8)
    ax_heat.clabel(contours, inline=True, fontsize=7, fmt="%d m")
    ax_heat.set_title("A. Coupled Thermal Propagation & Elevation Field", fontsize=11, fontweight="bold", pad=8, color="#F8FAFC")
    ax_heat.set_xlabel("X Spatial Coordinate (m)", fontsize=9)
    ax_heat.set_ylabel("Y Spatial Coordinate (m)", fontsize=9)
    
    # Panel B: Wind Vector Field & Combustion Front
    ax_wind = axes[0, 1]
    skip = 4
    ax_wind.contourf(env.X, env.Y, env.fuel, levels=12, cmap='Greens_r', alpha=0.55)
    quiv = ax_wind.quiver(
        env.X[::skip, ::skip],
        env.Y[::skip, ::skip],
        env.wind_u[::skip, ::skip],
        env.wind_v[::skip, ::skip],
        color="#38BDF8",
        alpha=0.85,
        scale=85
    )
    fire_perimeter = ax_wind.contour(
        env.X, env.Y, env.temperature,
        levels=[480, 750],
        colors=["#F59E0B", "#EF4444"],
        linewidths=[1.2, 1.8]
    )
    ax_wind.set_title("B. Dynamic Advection Vectors & Fuel Front", fontsize=11, fontweight="bold", pad=8, color="#F8FAFC")
    ax_wind.set_xlabel("X Spatial Coordinate (m)", fontsize=9)
    ax_wind.set_ylabel("Y Spatial Coordinate (m)", fontsize=9)
    
    # Panel C: Autonomous UAV Swarm Trajectories & Retardant Drops
    ax_swarm = axes[1, 0]
    ax_swarm.contourf(env.X, env.Y, env.retardant, levels=10, cmap='Purples', alpha=0.6)
    for drone in swarm.drones:
        traj = np.array(drone.trajectory)
        ax_swarm.plot(traj[:, 0], traj[:, 1], color="#06B6D4", alpha=0.5, linewidth=1.1)
        ax_swarm.scatter([drone.x], [drone.y], color="#A855F7", s=35, zorder=5, edgecolors="#FFFFFF")
        if drone.drops:
            drops = np.array(drone.drops)
            ax_swarm.scatter(drops[:, 0], drops[:, 1], color="#EC4899", s=15, marker="x", alpha=0.7)
    ax_swarm.set_title("C. Multi-Agent Drone Trajectories & Drop Zones", fontsize=11, fontweight="bold", pad=8, color="#F8FAFC")
    ax_swarm.set_xlabel("X Spatial Coordinate (m)", fontsize=9)
    ax_swarm.set_ylabel("Y Spatial Coordinate (m)", fontsize=9)
    
    # Panel D: PINN PDE Residual Loss & Containment Efficiency
    ax_loss = axes[1, 1]
    steps_arr = np.arange(1, len(containment_history) + 1)
    color_loss = "#F43F5E"
    ax_loss.plot(steps_arr, pinn.loss_history[:len(steps_arr)], color=color_loss, linewidth=1.8, label="PINN Energy Loss")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Simulation Timesteps (t)", fontsize=9)
    ax_loss.set_ylabel("Residual PDE Loss (Log Scale)", color=color_loss, fontsize=9)
    ax_loss.tick_params(axis='y', labelcolor=color_loss)
    
    ax_eff = ax_loss.twinx()
    color_eff = "#10B981"
    ax_eff.plot(steps_arr, containment_history, color=color_eff, linewidth=2.0, linestyle="--", label="Containment Rate (%)")
    ax_eff.set_ylabel("Perimeter Containment (%)", color=color_eff, fontsize=9)
    ax_eff.tick_params(axis='y', labelcolor=color_eff)
    ax_loss.set_title("D. Neural Operator Convergence & Swarm Efficacy", fontsize=11, fontweight="bold", pad=8, color="#F8FAFC")
    
    plt.savefig("assets/aeropyre_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[✓] AeroPyre-Sim visualization saved to assets/aeropyre_dashboard.png")

if __name__ == "__main__":
    run_simulation()
