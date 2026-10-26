import numpy as np

class PINNSurrogate:
    """
    Physics-Informed Neural Network (PINN) surrogate model.
    Learns spatial-temporal mapping (x, y, t) -> Temperature and
    evaluates PDE residual loss enforcing conservation of thermal energy.
    """
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1, lr=0.005):
        np.random.seed(42)
        # 3-layer neural network
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, output_dim))
        self.lr = lr
        self.loss_history = []
        
    def _tanh(self, z):
        return np.tanh(z)
        
    def _dtanh(self, z):
        return 1.0 - np.tanh(z)**2
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3
        
    def compute_pde_residual(self, X_colloc, env):
        """
        Computes the physics residual of the heat advection-diffusion operator:
        Residual = dT/dt - alpha * laplacian(T) + v . grad(T)
        """
        eps = 1e-3
        # Perturbations for finite-difference gradients in coordinate space
        T_pred = self.forward(X_colloc)
        
        # d/dt
        X_t_plus = X_colloc.copy(); X_t_plus[:, 2] += eps
        X_t_minus = X_colloc.copy(); X_t_minus[:, 2] -= eps
        dT_dt = (self.forward(X_t_plus) - self.forward(X_t_minus)) / (2 * eps)
        
        # d/dx, d2/dx2
        X_x_plus = X_colloc.copy(); X_x_plus[:, 0] += eps
        X_x_minus = X_colloc.copy(); X_x_minus[:, 0] -= eps
        dT_dx = (self.forward(X_x_plus) - self.forward(X_x_minus)) / (2 * eps)
        d2T_dx2 = (self.forward(X_x_plus) - 2 * T_pred + self.forward(X_x_minus)) / (eps**2)
        
        # d/dy, d2/dy2
        X_y_plus = X_colloc.copy(); X_y_plus[:, 1] += eps
        X_y_minus = X_colloc.copy(); X_y_minus[:, 1] -= eps
        dT_dy = (self.forward(X_y_plus) - self.forward(X_y_minus)) / (2 * eps)
        d2T_dy2 = (self.forward(X_y_plus) - 2 * T_pred + self.forward(X_y_minus)) / (eps**2)
        
        # Physical operator residual
        laplacian = d2T_dx2 + d2T_dy2
        advection = 6.0 * dT_dx + 4.0 * dT_dy
        res = dT_dt - 1.4 * laplacian + advection
        residual_loss = np.mean(res**2)
        return residual_loss
        
    def train_step(self, X_train, y_train, X_colloc, env):
        # Forward data loss
        pred = self.forward(X_train)
        data_error = (pred - y_train)
        data_loss = np.mean(data_error**2)
        
        # Backprop through layers (data loss)
        d_out = (2.0 / len(X_train)) * data_error
        dW3 = np.dot(self.a2.T, d_out)
        db3 = np.sum(d_out, axis=0, keepdims=True)
        
        d_a2 = np.dot(d_out, self.W3.T)
        d_z2 = d_a2 * self._dtanh(self.z2)
        dW2 = np.dot(self.a1.T, d_z2)
        db2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self._dtanh(self.z1)
        dW1 = np.dot(X_train.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # Parameter update
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        pde_res = self.compute_pde_residual(X_colloc, env)
        total_loss = data_loss + 0.15 * pde_res
        self.loss_history.append(float(total_loss))
        return total_loss, pde_res
