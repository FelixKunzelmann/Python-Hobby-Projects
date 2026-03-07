"""
Quantum Tunneling Simulator
Simulates a quantum particle tunneling through a potential barrier using
the time-dependent Schrödinger equation and visualizes the result.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


class QuantumTunnelSimulator:
    """Simulates quantum particle tunneling through a barrier."""

    def __init__(self, x_min=-30, x_max=30, num_points=1024,
                 barrier_height=3.0, barrier_width=2.0, barrier_center=0.0,
                 dt=0.01, t_max=20):
        """
        Initialize the quantum tunneling simulator.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.num_points = num_points
        self.dt = dt
        self.t_max = t_max

        # spatial grid
        self.x = np.linspace(x_min, x_max, num_points)
        self.dx = (x_max - x_min) / (num_points - 1)

        # potential barrier
        self.potential = self._create_barrier(
            barrier_height, barrier_width, barrier_center)

        # initial Gaussian packet
        self.psi = self._create_gaussian_packet(x0=-5, sigma=0.5, k0=3.0)

        # kinetic operator
        self.T_operator = self._create_kinetic_operator()

        # storage
        self.times = []
        self.psi_history = []
        self.prob_history = []

    def _create_barrier(self, height, width, center):
        barrier = np.zeros_like(self.x)
        mask = np.abs(self.x - center) <= width / 2
        barrier[mask] = height
        return barrier

    def _create_gaussian_packet(self, x0=-5, sigma=0.5, k0=5.0):
        envelope = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        wave = np.exp(1j * k0 * self.x)
        psi = envelope * wave
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        return psi

    # The discretized kinetic energy operator using finite difference method. The value of the wavefunction at the boundaries is therfore set to zero, which results in seaming refections at the boundaries for too long simulation times. This is an unphysical effect.
    def _create_kinetic_operator(self):
        diag_main = -2.0 / (self.dx**2) * np.ones(self.num_points)
        diag_off = 1.0 / (self.dx**2) * np.ones(self.num_points - 1)
        T = diags([diag_main, diag_off, diag_off], [0, 1, -1],  # type: ignore
                  shape=(self.num_points, self.num_points), format='csr')
        T *= -0.5
        return T

    def step(self):
        V_operator = diags(self.potential, 0, format='csr')
        psi_half = np.exp(-1j * self.potential * self.dt / 2) * self.psi
        total = self.T_operator + V_operator
        psi_full = expm_multiply(total * (-1j * self.dt / 2), psi_half)
        self.psi = np.exp(-1j * self.potential * self.dt / 2) * psi_full
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        self.psi /= norm

    def run_simulation(self):
        num_steps = int(self.t_max / self.dt)
        print("Running quantum tunneling simulation...")
        print(f"Simulation time: {self.t_max} time units")
        print(f"Number of steps: {num_steps}")
        for step_num in range(num_steps):
            if step_num % max(1, num_steps // 20) == 0:
                print(
                    f"  Progress: {step_num}/{num_steps} ({100*step_num/num_steps:.1f}%)")
            self.step()
            if step_num % 2 == 0:
                self.times.append(step_num * self.dt)
                self.psi_history.append(self.psi.copy())
                self.prob_history.append(np.abs(self.psi)**2)
        print("Calculation complete!")

    def calculate_transmission(self):
        barrier_center = np.mean(self.x[np.abs(self.potential) > 0.1])
        right_side = self.x > barrier_center
        transmissions = []
        for prob in self.prob_history:
            transmissions.append(np.sum(prob[right_side]) * self.dx)
        return np.array(transmissions)

    def calculate_reflection(self):
        barrier_center = np.mean(self.x[np.abs(self.potential) > 0.1])
        left_side = self.x < barrier_center
        reflections = []
        for prob in self.prob_history:
            reflections.append(np.sum(prob[left_side]) * self.dx)
        return np.array(reflections)

    def plot_static_overview(self):
        # layout: 2 rows × 2 cols; Totoal probability plot not shown here
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quantum Tunneling Simulation Overview',
                     fontsize=16, fontweight='bold')

        ax = axes[0, 0]
        ax.fill_between(self.x, np.abs(
            self.psi_history[0])**2, alpha=0.7, color='blue')
        ax.plot(self.x, self.potential, 'r-',
                linewidth=2, label='Potential Barrier')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Probability Density |ψ|²')
        ax.set_title('Initial State (t=0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        reflection = self.calculate_reflection()
        ax.plot(self.times, reflection, 'r-', linewidth=2)
        ax.fill_between(self.times, reflection, alpha=0.3)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Probability of finding particle on the left side')
        ax.set_title('Reflection Probability Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        ax = axes[1, 0]
        transmissions = self.calculate_transmission()
        ax.plot(self.times, transmissions, 'b-', linewidth=2)
        ax.fill_between(self.times, transmissions, alpha=0.3)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Probability of finding particle on the right side')
        ax.set_title('Tunneling Probability Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        ax = axes[1, 1]
        prob_array = np.array([np.abs(psi)**2 for psi in self.psi_history])
        im = ax.contourf(self.x, self.times, prob_array,
                         levels=np.arange(0, 1, 0.01), cmap='viridis', vmax=1)
        ax.plot(self.x, -0.1+self.potential *
                (self.times[-1]-self.times[0]+1), 'r-', linewidth=2)
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Time (t)')
        ax.set_title('Probability Density Evolution')
        ax.set_xlim(-10, 10)
        ax.set_ylim((self.times[0], self.times[-1]))
        cbar = plt.colorbar(im, ax=ax, label='|ψ|²')

        # total probability plot
        # ax = axes[2, 0]
        # total_prob = [np.sum(prob) * self.dx for prob in self.prob_history]
        # ax.plot(self.times, total_prob, 'k-', linewidth=2)
        # ax.set_xlabel('Time (t)')
        # ax.set_ylabel('Total Probability')
        # ax.set_title('Normalization: ∫|ψ|² dx over time')
        # ax.grid(True, alpha=0.3)
        # ax.set_ylim([min(total_prob) * 0.99, max(total_prob) * 1.01])

        # hide unused subplot
        # axes[2, 1].axis('off')

        plt.tight_layout()
        # plt.show()  # Remove blocking show

    def create_animation(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('Quantum Particle Tunneling Through Barrier',
                     fontsize=14, fontweight='bold')

        ax1.set_xlim(self.x_min, self.x_max)
        max_prob = np.array(self.psi_history)
        ax1.set_ylim(0, np.abs(max_prob).max()**2 * 1.1)
        ax1.set_xlabel('Position (x)')
        ax1.set_ylabel('Probability Density |ψ|²')
        ax1.grid(True, alpha=0.3)
        ax1.plot(self.x, self.potential, 'r-', linewidth=2.5)
        prob_line, = ax1.plot(self.x, np.abs(
            self.psi_history[0])**2, 'b-', linewidth=1.5)

        ax2.set_xlim(self.times[0], self.times[-1])
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time (t)')
        ax2.set_ylabel('Transmission Prob.')
        ax2.grid(True, alpha=0.3)
        transmission_line, = ax2.plot([], [], 'g-', linewidth=2)

        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        transmissions = self.calculate_transmission()

        def animate(frame):
            prob = np.abs(self.psi_history[frame])**2
            prob_line.set_ydata(prob)
            current_times = self.times[:frame+1]
            current_trans = transmissions[:frame+1]
            transmission_line.set_data(current_times, current_trans)
            time_text.set_text(
                f'Time: {self.times[frame]:.2f}\nTransmission: {transmissions[frame]:.3f}')
            return prob_line, transmission_line, time_text

        ani = FuncAnimation(fig, animate, frames=len(self.psi_history),
                            interval=50, blit=False, repeat=True)
        if save_path:
            ani.save(save_path, dpi=100)
        # plt.show()  # Remove blocking show
        return ani


def main():
    print("\n" + "="*60)
    print("QUANTUM TUNNELING SIMULATOR")
    print("="*60 + "\n")
    simulator = QuantumTunnelSimulator(
        x_min=-30, x_max=50,
        num_points=1024,
        barrier_height=1.2,
        barrier_width=2.0,
        barrier_center=0.0,
        dt=0.01,
        t_max=20
    )
    simulator.run_simulation()
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print("\nGenerating static overview plot...")
    simulator.plot_static_overview()
    print("\nGenerating animation...")
    ani = simulator.create_animation()
    print("\nDisplaying plots...")
    plt.show()  # Show both figures at once
    print("\n" + "="*60)
    print("Calculation completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
