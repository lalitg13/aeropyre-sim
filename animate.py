import os
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from src.physics import WildfireEnvironment
from src.swarm import DroneSwarm

def generate_spinning_animation():
    print("[*] Generating 3D Spinning Wildfire & Swarm Defense Animation for AeroPyre-Sim...")
    env = WildfireEnvironment(grid_size=45, dx=13.0, dt=0.6)
    swarm = DroneSwarm(num_drones=10, env_width=45 * 13.0)
    
    # Pre-advance simulation 50 steps so fire front is active across the mountain
    for _ in range(50):
        env.step()
        swarm.update(env)
        
    frames = []
    total_frames = 36
    
    # Downsample grid for fast rendering
    X_3d = env.X
    Y_3d = env.Y
    Z_3d = env.elevation
    
    # Normalize temperature for 3D surface facecolors
    temp_norm = np.clip((env.temperature - 295.0) / 750.0, 0.0, 1.0)
    cmap = plt.get_cmap("inferno")
    facecolors = cmap(temp_norm)
    
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(7, 6), dpi=120)
    
    for i, azim in enumerate(np.linspace(0, 360, total_frames, endpoint=False)):
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#030712")
        fig.patch.set_facecolor("#030712")
        
        # 3D Mountain Terrain with Inferno Flame Front
        surf = ax.plot_surface(
            X_3d, Y_3d, Z_3d,
            facecolors=facecolors,
            shade=True,
            linewidth=0.15,
            edgecolor="#1E293B",
            alpha=0.95
        )
        
        # Plot Drone Swarm hovering above terrain
        for drone in swarm.drones:
            # Interpolate ground elevation
            gi = int(np.clip(drone.y / env.dx, 0, env.N - 1))
            gj = int(np.clip(drone.x / env.dx, 0, env.N - 1))
            alt = env.elevation[gi, gj] + 45.0  # 45m flight altitude
            ax.scatter([drone.x], [drone.y], [alt], color="#06B6D4", s=35, edgecolors="#FFFFFF", linewidths=1.0)
            
        elev = 32.0 + 8.0 * np.sin(np.radians(azim * 2))
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.axis("off")
        ax.set_title("3D Topographical Wildfire & UAV Swarm Defense", color="#F8FAFC", fontsize=11, fontweight="bold", pad=-5)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#030712", pad_inches=0.1)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        
    plt.close()
    
    output_path = "assets/aeropyre_3d_spin.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=65,
        loop=0
    )
    print(f"[✓] Saved 3D spinning wildfire swarm animation to {output_path}")

if __name__ == "__main__":
    generate_spinning_animation()
