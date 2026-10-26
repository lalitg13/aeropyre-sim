import numpy as np

class WildfireEnvironment:
    """
    2D Coupled Reaction-Diffusion and Convective Thermal Transport Model
    for Wildfire Propagation over complex terrain with wind fields.
    """
    def __init__(self, grid_size=60, dx=10.0, dt=0.5):
        self.N = grid_size
        self.dx = dx
        self.dt = dt
        
        # Spatial coordinates (meters)
        x = np.linspace(0, (self.N - 1) * self.dx, self.N)
        y = np.linspace(0, (self.N - 1) * self.dx, self.N)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Elevation profile (Gaussian hills & valleys)
        self.elevation = (
            250 * np.exp(-((self.X - 250)**2 + (self.Y - 250)**2) / (2 * 120**2)) +
            180 * np.exp(-((self.X - 420)**2 + (self.Y - 380)**2) / (2 * 90**2))
        )
        # Terrain gradient (slope effects on fire speed)
        self.grad_elev_y, self.grad_elev_x = np.gradient(self.elevation, self.dx)
        
        # Atmospheric wind field (m/s) with localized turbulence
        base_wind_u = 6.5
        base_wind_v = 4.0
        self.wind_u = base_wind_u + 1.2 * np.sin(self.Y / 80.0)
        self.wind_v = base_wind_v + 1.2 * np.cos(self.X / 80.0)
        
        # State variables
        self.temperature = np.full((self.N, self.N), 295.0, dtype=np.float64)  # Kelvin (ambient ~22C)
        self.fuel = np.ones((self.N, self.N), dtype=np.float64)                # Fuel density [0, 1]
        self.retardant = np.zeros((self.N, self.N), dtype=np.float64)           # Retardant concentration [0, 1]
        
        # Ignition source at (x=15, y=15)
        self.ignite(15, 15, radius=4, peak_temp=1100.0)
        
        # Physical constants
        self.thermal_diffusivity = 1.4  # m^2/s
        self.ambient_temp = 295.0       # K
        self.combustion_energy = 850.0  # J/kg equivalent
        self.ignition_thresh = 480.0    # K
        self.cooling_coeff = 0.045
        
    def ignite(self, cx, cy, radius=3, peak_temp=1150.0):
        for i in range(max(0, cy - radius), min(self.N, cy + radius + 1)):
            for j in range(max(0, cx - radius), min(self.N, cx + radius + 1)):
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                if dist <= radius:
                    self.temperature[i, j] = peak_temp * np.exp(-dist / 2.0)
                    self.fuel[i, j] = 0.85
                    
    def step(self):
        """Advances physics by dt using finite difference scheme."""
        T = self.temperature
        F = self.fuel
        R = self.retardant
        
        # 5-point discrete Laplacian for thermal conduction
        laplacian_T = (
            np.roll(T, -1, axis=0) + np.roll(T, 1, axis=0) +
            np.roll(T, -1, axis=1) + np.roll(T, 1, axis=1) - 4 * T
        ) / (self.dx ** 2)
        
        # Upwind finite differences for wind advection
        dT_dx = np.where(self.wind_u > 0, (T - np.roll(T, 1, axis=1)) / self.dx, (np.roll(T, -1, axis=1) - T) / self.dx)
        dT_dy = np.where(self.wind_v > 0, (T - np.roll(T, 1, axis=0)) / self.dx, (np.roll(T, -1, axis=0) - T) / self.dx)
        advection = self.wind_u * dT_dx + self.wind_v * dT_dy
        
        # Slope factor (fire climbs uphill significantly faster)
        slope_factor = 1.0 + 2.0 * np.maximum(0, (self.wind_u * self.grad_elev_x + self.wind_v * self.grad_elev_y) / (np.linalg.norm([self.wind_u, self.wind_v]) + 1e-5))
        
        # Combustion reaction rate (Arrhenius-like threshold)
        active_burn = (T > self.ignition_thresh) & (F > 0.05)
        burn_rate = np.zeros_like(T)
        burn_rate[active_burn] = 0.08 * (1.0 - R[active_burn]) * F[active_burn] * np.minimum((T[active_burn] - self.ignition_thresh) / 100.0, 3.0)
        
        # Fuel consumption
        self.fuel = np.clip(self.fuel - burn_rate * self.dt, 0.0, 1.0)
        
        # Thermal generation, cooling, and retardant quenching
        heat_release = burn_rate * self.combustion_energy * slope_factor
        cooling = self.cooling_coeff * (T - self.ambient_temp)
        quench = 0.15 * R * (T - self.ambient_temp)
        
        # Coupled PDE Update
        dT_dt = self.thermal_diffusivity * laplacian_T - advection + heat_release - cooling - quench
        self.temperature = np.clip(T + dT_dt * self.dt, self.ambient_temp, 1600.0)
        
        # Retardant decay
        self.retardant = np.clip(self.retardant * 0.998, 0.0, 1.0)
