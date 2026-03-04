import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable

# Prefer the community edition package `pygame_ce` on newer Python versions,
# but fall back to the official `pygame` package when available.
PYGAME_AVAILABLE = False
try:
    import pygame_ce as pygame  # type: ignore
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import pygame  # type: ignore
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False


@dataclass
class Vector2D:
    """Simple 2D vector class for position and velocity."""
    x: float
    y: float

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def dot(self, other: 'Vector2D') -> float:
        return self.x * other.x + self.y * other.y


class Particle:
    """Represents a gas molecule."""

    def __init__(self, x: float, y: float, vx: float, vy: float, radius: float = 1.0, mass: float = 1.0):
        """
        Initialize a particle.

        Args:
            x, y: Initial position
            vx, vy: Initial velocity components
            radius: Particle radius (for collision detection)
            mass: Particle mass
        """
        self.pos = Vector2D(x, y)
        self.vel = Vector2D(vx, vy)
        self.radius = radius
        self.mass = mass

    def update(self, dt: float):
        """Update particle position based on velocity."""
        self.pos = self.pos + self.vel * dt

    def kinetic_energy(self) -> float:
        """Calculate kinetic energy of the particle."""
        v_squared = self.vel.x**2 + self.vel.y**2
        return 0.5 * self.mass * v_squared

    def speed(self) -> float:
        """Get particle speed."""
        return self.vel.magnitude()


class RectangularBox:
    """Container for the ideal gas."""

    def __init__(self, width: float, height: float):
        """
        Initialize rectangular box.

        Args:
            width: Width of the box
            height: Height of the box
        """
        self.width = width
        self.height = height
        self.volume = width * height
        self.wall_collisions = 0
        self.particle_collisions = 0

    def is_inside(self, particle: Particle) -> bool:
        """Check if particle is inside the box (with margin for radius)."""
        return (particle.radius <= particle.pos.x <= self.width - particle.radius and
                particle.radius <= particle.pos.y <= self.height - particle.radius)

    def handle_wall_collision(self, particle: Particle):
        """Handle collision between particle and walls (elastic collision)."""
        # Check right and left walls
        if particle.pos.x - particle.radius <= 0:
            particle.pos.x = particle.radius
            particle.vel.x = abs(particle.vel.x)
            self.wall_collisions += 1
        elif particle.pos.x + particle.radius >= self.width:
            particle.pos.x = self.width - particle.radius
            particle.vel.x = -abs(particle.vel.x)
            self.wall_collisions += 1

        # Check top and bottom walls
        if particle.pos.y - particle.radius <= 0:
            particle.pos.y = particle.radius
            particle.vel.y = abs(particle.vel.y)
            self.wall_collisions += 1
        elif particle.pos.y + particle.radius >= self.height:
            particle.pos.y = self.height - particle.radius
            particle.vel.y = -abs(particle.vel.y)
            self.wall_collisions += 1

    def distance_between(self, p1: Particle, p2: Particle) -> float:
        """Calculate distance between two particles."""
        dx = p2.pos.x - p1.pos.x
        dy = p2.pos.y - p1.pos.y
        return math.sqrt(dx**2 + dy**2)

    def handle_particle_collision(self, p1: Particle, p2: Particle):
        """Handle elastic collision between two particles."""
        distance = self.distance_between(p1, p2)
        min_distance = p1.radius + p2.radius

        if distance < min_distance:
            # Normal vector
            nx = (p2.pos.x - p1.pos.x) / distance
            ny = (p2.pos.y - p1.pos.y) / distance

            # Relative velocity
            dvx = p2.vel.x - p1.vel.x
            dvy = p2.vel.y - p1.vel.y

            # Relative velocity along normal
            dvn = dvx * nx + dvy * ny

            # Only collide if particles are moving towards each other
            if dvn < 0:
                # Impulse scalar
                impulse = (2 * dvn) / (p1.mass + p2.mass)

                # Update velocities
                p1.vel.x += impulse * p2.mass * nx
                p1.vel.y += impulse * p2.mass * ny
                p2.vel.x -= impulse * p1.mass * nx
                p2.vel.y -= impulse * p1.mass * ny

                # Separate particles to avoid overlap
                overlap = min_distance - distance
                separation = overlap / 2 + 0.01
                p1.pos.x -= separation * nx
                p1.pos.y -= separation * ny
                p2.pos.x += separation * nx
                p2.pos.y += separation * ny

                self.particle_collisions += 1


class IdealGasSimulation:
    """Simulates an ideal gas in a rectangular box."""

    def __init__(self, box_width: float, box_height: float, num_particles: int,
                 particle_radius: float = 1.0, particle_mass: float = 1.0,
                 temperature: float = 1.0):
        """
        Initialize the simulation.

        Args:
            box_width: Width of the container
            box_height: Height of the container
            num_particles: Number of gas particles
            particle_radius: Radius of each particle
            particle_mass: Mass of each particle
            temperature: Temperature (affects initial velocities)
        """
        self.box = RectangularBox(box_width, box_height)
        self.particles: List[Particle] = []
        self.time = 0.0
        self.temperature = temperature

        # Create particles with random positions and velocities
        self._initialize_particles(
            num_particles, particle_radius, particle_mass)

    def _initialize_particles(self, num_particles: int, radius: float, mass: float):
        """Initialize particles with random positions and velocities."""
        for _ in range(num_particles):
            # Random position within box
            x = random.uniform(radius, self.box.width - radius)
            y = random.uniform(radius, self.box.height - radius)

            # Random velocity based on temperature (Maxwell-Boltzmann distribution)
            # v ~ sqrt(kT/m), we simplify by using temperature directly
            speed = math.sqrt(2 * self.temperature)
            angle = random.uniform(0, 2 * math.pi)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            particle = Particle(x, y, vx, vy, radius, mass)
            self.particles.append(particle)

    def step(self, dt: float = 0.01, handle_collisions: bool = True):
        """
        Perform one simulation step.

        Args:
            dt: Time step
            handle_collisions: Whether to handle particle-particle collisions
        """
        # Update positions
        for particle in self.particles:
            particle.update(dt)

        # Handle wall collisions
        for particle in self.particles:
            self.box.handle_wall_collision(particle)

        # Handle particle collisions
        if handle_collisions:
            for i in range(len(self.particles)):
                for j in range(i + 1, len(self.particles)):
                    self.box.handle_particle_collision(
                        self.particles[i], self.particles[j])

        self.time += dt

    def get_average_speed(self) -> float:
        """Calculate average speed of particles."""
        if not self.particles:
            return 0.0
        total_speed = sum(p.speed() for p in self.particles)
        return total_speed / len(self.particles)

    def get_total_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of all particles."""
        return sum(p.kinetic_energy() for p in self.particles)

    def get_pressure(self) -> float:
        """
        Estimate pressure using ideal gas law approximation.
        P = N*k*T/V, where we use wall collisions as proxy for pressure.
        """
        if self.box.volume == 0:
            return 0.0
        # Pressure proportional to kinetic energy and inversely proportional to volume
        kinetic_energy = self.get_total_kinetic_energy()
        return (2 * kinetic_energy) / (3 * self.box.volume)

    def get_statistics(self) -> dict:
        """Get current simulation statistics."""
        return {
            'time': self.time,
            'num_particles': len(self.particles),
            'average_speed': self.get_average_speed(),
            'total_kinetic_energy': self.get_total_kinetic_energy(),
            'pressure': self.get_pressure(),
            'wall_collisions': self.box.wall_collisions,
            'particle_collisions': self.box.particle_collisions,
            'temperature': self.temperature,
            'volume': self.box.volume
        }


class Slider:
    """Interactive slider control for adjusting parameters."""

    def __init__(self, x: int, y: int, width: int, height: int, min_val: float,
                 max_val: float, initial_val: float, label: str, on_change: Callable[[float], None] = None):
        """
        Initialize slider.

        Args:
            x, y: Position on screen
            width, height: Dimensions of slider
            min_val: Minimum value
            max_val: Maximum value
            initial_val: Initial value
            label: Label for the slider
            on_change: Callback function when slider value changes
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.on_change = on_change
        self.dragging = False

        # Calculate knob position
        self._update_knob_pos()

    def _update_knob_pos(self):
        """Update knob position based on current value."""
        ratio = (self.value - self.min_val) / \
            max(self.max_val - self.min_val, 0.0001)
        self.knob_x = self.x + int(ratio * self.width)
        self.knob_y = self.y + self.height // 2

    def get_rect(self) -> pygame.Rect:
        """Get slider track rectangle."""
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def get_knob_rect(self) -> pygame.Rect:
        """Get knob rectangle."""
        knob_size = 12
        return pygame.Rect(self.knob_x - knob_size // 2, self.knob_y - knob_size // 2,
                           knob_size, knob_size)

    def handle_mouse_down(self, pos: Tuple[int, int]) -> bool:
        """Check if mouse clicked on slider knob."""
        if self.get_knob_rect().collidepoint(pos):
            self.dragging = True
            return True
        return False

    def handle_mouse_up(self):
        """Stop dragging."""
        self.dragging = False

    def handle_mouse_move(self, pos: Tuple[int, int]):
        """Update slider value based on mouse position."""
        if self.dragging:
            mouse_x = pos[0]
            # Clamp mouse_x to slider bounds
            mouse_x = max(self.x, min(mouse_x, self.x + self.width))

            # Calculate new value
            ratio = (mouse_x - self.x) / max(self.width, 1)
            new_value = self.min_val + ratio * (self.max_val - self.min_val)

            # Update if value changed
            if abs(new_value - self.value) > 0.001:
                self.value = new_value
                self._update_knob_pos()
                if self.on_change:
                    self.on_change(self.value)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, colors: dict):
        """Draw slider on screen."""
        # Draw label
        label_text = font.render(
            f"{self.label}: {self.value:.2f}", True, colors['WHITE'])
        screen.blit(label_text, (self.x, self.y - 25))

        # Draw track
        pygame.draw.rect(screen, colors['GRAY'], self.get_rect(), 2)

        # Draw knob
        pygame.draw.circle(
            screen, colors['GRAY'], (self.knob_x, self.knob_y), 8)
        if self.dragging:
            pygame.draw.circle(
                screen, colors['WHITE'], (self.knob_x, self.knob_y), 8, 2)


class GasVisualizer:
    """Visualizes the ideal gas simulation using pygame."""

    def __init__(self, simulation: IdealGasSimulation, window_width: int = 1000,
                 window_height: int = 800, fps: int = 60):
        """
        Initialize the visualizer.

        Args:
            simulation: IdealGasSimulation instance to visualize
            window_width: Width of display window
            window_height: Height of display window
            fps: Frames per second for visualization
        """
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for visualization. Install with: pip install pygame")

        self.sim = simulation
        self.window_width = window_width
        self.window_height = window_height
        self.fps = fps

        # Calculate scaling factors (reserve space for controls at bottom)
        self.scale_x = (window_width - 100) / simulation.box.width
        self.scale_y = (window_height - 250) / simulation.box.height
        self.offset_x = 50
        self.offset_y = 50

        # Control panel area starts at this y coordinate
        self.control_area_y = 25 + \
            int(simulation.box.height * self.scale_y) + 20

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (100, 100, 100)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)
        self.GREEN = (0, 255, 0)
        self.CYAN = (0, 255, 255)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Ideal Gas Simulation")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)

        # Create sliders in the control area
        self.temp_slider = Slider(
            x=window_width - 400, y=self.control_area_y + 40, width=250, height=20,
            min_val=10.0, max_val=5000.0,
            initial_val=simulation.temperature,
            label="Temperature",
            on_change=self.on_temperature_change
        )

        self.particle_slider = Slider(
            x=window_width - 400, y=self.control_area_y + 40 + 60, width=250, height=20,
            min_val=10, max_val=1000,
            initial_val=len(simulation.particles),
            label="Particles",
            on_change=self.on_particle_count_change
        )

        self.running = True
        self.paused = False

    def get_screen_pos(self, x: float, y: float) -> Tuple[int, int]:
        """Convert simulation coordinates to screen coordinates."""
        screen_x = int(x * self.scale_x + self.offset_x)
        screen_y = int(y * self.scale_y + self.offset_y)
        return (screen_x, screen_y)

    def get_particle_color(self, particle: Particle) -> Tuple[int, int, int]:
        """Get color based on particle speed (red=slow, blue=fast)."""
        # Scale colors relative to the thermal speed derived from temperature.
        # Initial particle speeds are generated with `sqrt(2 * temperature)`.
        thermal_speed = math.sqrt(max(self.sim.temperature, 0.0) * 2)
        max_speed = max(thermal_speed * 3.0, 0.1)
        speed = min(particle.speed() / max_speed, 1.0)

        # Gradient from red to blue
        red = int(255 * (1 - speed))
        blue = int(255 * speed)
        green = 50

        return (red, green, blue)

    def on_temperature_change(self, new_temp: float):
        """Callback when temperature slider changes."""
        self.sim.temperature = new_temp
        # Adjust velocities of all particles to match new temperature
        thermal_speed = math.sqrt(2 * new_temp)
        for particle in self.sim.particles:
            current_speed = particle.speed()
            if current_speed > 0.001:
                # Scale velocity to new thermal speed
                ratio = current_speed / current_speed if current_speed > 0 else 1.0
                scale_factor = thermal_speed / current_speed if current_speed > 0 else 1.0
                particle.vel.x *= scale_factor
                particle.vel.y *= scale_factor

    def on_particle_count_change(self, new_count: float):
        """Callback when particle count slider changes."""
        new_count = int(new_count)
        current_count = len(self.sim.particles)

        if new_count > current_count:
            # Add new particles
            particles_to_add = new_count - current_count
            for _ in range(particles_to_add):
                x = random.uniform(
                    self.sim.particles[0].radius if self.sim.particles else 1.0,
                    self.sim.box.width -
                    (self.sim.particles[0].radius if self.sim.particles else 1.0)
                )
                y = random.uniform(
                    self.sim.particles[0].radius if self.sim.particles else 1.0,
                    self.sim.box.height -
                    (self.sim.particles[0].radius if self.sim.particles else 1.0)
                )
                speed = math.sqrt(2 * self.sim.temperature)
                angle = random.uniform(0, 2 * math.pi)
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)

                particle_radius = self.sim.particles[0].radius if self.sim.particles else 1.0
                particle_mass = self.sim.particles[0].mass if self.sim.particles else 1.0

                new_particle = Particle(
                    x, y, vx, vy, particle_radius, particle_mass)
                self.sim.particles.append(new_particle)

        elif new_count < current_count:
            # Remove particles from the end
            particles_to_remove = current_count - new_count
            for _ in range(particles_to_remove):
                self.sim.particles.pop()

    def draw_box(self):
        """Draw the simulation box."""
        x1, y1 = self.get_screen_pos(0, 0)
        x2, y2 = self.get_screen_pos(self.sim.box.width, self.sim.box.height)

        # Draw box outline
        pygame.draw.rect(self.screen, self.WHITE,
                         (x1, y1, x2 - x1, y2 - y1), 3)

    def draw_particles(self):
        """Draw all particles."""
        for particle in self.sim.particles:
            x, y = self.get_screen_pos(particle.pos.x, particle.pos.y)
            radius = max(int(particle.radius * self.scale_x), 3)
            color = self.get_particle_color(particle)

            pygame.draw.circle(self.screen, color, (x, y), radius)

    def draw_statistics(self):
        """Draw simulation statistics and controls on screen."""
        stats = self.sim.get_statistics()

        left_margin = 10

        # Title at top of control area
        title = self.font_large.render(
            "Ideal Gas Simulation", True, self.WHITE)
        self.screen.blit(title, (left_margin, 10))

        # Statistics on the left side of control area
        stats_y = self.control_area_y + 20
        stat_texts = [
            f"Time: {stats['time']:.2f}s",
            f"Particles: {stats['num_particles']}  |  Avg Speed: {stats['average_speed']:.2f}",
            f"Temperature: {stats['temperature']:.2f}  |  Pressure: {stats['pressure']:.4f}",
            f"Total KE: {stats['total_kinetic_energy']:.2f}  |  Wall Collisions: {stats['wall_collisions']}"
        ]

        for text in stat_texts:
            label = self.font_small.render(text, True, self.WHITE)
            self.screen.blit(label, (left_margin, stats_y))
            stats_y += 25

        # Mode indicator on the right
        mode_text = "PAUSED" if self.paused else "RUNNING"
        mode_color = self.RED if self.paused else self.GREEN
        mode = self.font_small.render(mode_text, True, mode_color)
        self.screen.blit(mode, (self.window_width -
                         120, self.control_area_y + 20))

        # Instructions at the very bottom
        instructions = self.font_small.render(
            "SPACE: Pause/Resume  |  Q: Quit", True, self.GRAY)
        self.screen.blit(instructions, (left_margin, self.window_height - 25))

        # Draw sliders with color dict
        colors = {
            'WHITE': self.WHITE,
            'GRAY': self.GRAY,
            'CYAN': self.CYAN,
            'BLACK': self.BLACK,
            'RED': self.RED,
            'BLUE': self.BLUE,
            'GREEN': self.GREEN
        }
        self.temp_slider.draw(self.screen, self.font_small, colors)
        self.particle_slider.draw(self.screen, self.font_small, colors)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_q:
                    self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                self.temp_slider.handle_mouse_down(pos)
                self.particle_slider.handle_mouse_down(pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.temp_slider.handle_mouse_up()
                self.particle_slider.handle_mouse_up()
            elif event.type == pygame.MOUSEMOTION:
                pos = pygame.mouse.get_pos()
                self.temp_slider.handle_mouse_move(pos)
                self.particle_slider.handle_mouse_move(pos)

    def run(self, max_steps: int = None):
        """
        Run the visualization.

        Args:
            max_steps: Maximum number of simulation steps (None for infinite)
        """
        step = 0

        while self.running:
            self.handle_events()

            # Update simulation if not paused
            if not self.paused:
                self.sim.step(dt=0.01, handle_collisions=True)
                step += 1

                if max_steps and step >= max_steps:
                    self.running = False

            # Draw
            self.screen.fill(self.BLACK)
            self.draw_box()
            self.draw_particles()
            self.draw_statistics()

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        print(f"\nSimulation ended after {step} steps")
        print(f"Final time: {self.sim.time:.2f}s")
        final_stats = self.sim.get_statistics()
        print(f"Final pressure: {final_stats['pressure']:.4f}")
        print(f"Final average speed: {final_stats['average_speed']:.2f}")


def main():
    """Run a demonstration of the ideal gas simulation."""
    print("=" * 60)
    print("Ideal Gas Simulation")
    print("=" * 60)

    # Ask user for mode
    if PYGAME_AVAILABLE:
        print("\nSelect mode:")
        print("  1. Visual (pygame) - Interactive visualization")
        print("  2. Console - Text output only")
        choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    else:
        print("\nNote: pygame not installed. Using console mode.")
        print("Install pygame with: pip install pygame")
        choice = "2"

    # Create simulation
    width, height = 100.0, 80.0
    num_particles = 500
    sim = IdealGasSimulation(width, height, num_particles,
                             particle_radius=0.5,
                             particle_mass=1.0,
                             temperature=1000.0)

    print(f"\nInitial conditions:")
    print(f"  Box dimensions: {width} x {height}")
    print(f"  Number of particles: {num_particles}")
    print(f"  Temperature: {sim.temperature}")
    print()

    if choice == "1" and PYGAME_AVAILABLE:
        # Visual mode
        print("Starting visual simulation...")
        visualizer = GasVisualizer(sim, window_width=1000, window_height=800)
        visualizer.run(max_steps=2000)
    else:
        # Console mode
        num_steps = 500
        print(f"Running simulation for {num_steps} steps...\n")

        for step in range(num_steps):
            sim.step(dt=0.01, handle_collisions=True)

            # Print statistics periodically
            if (step + 1) % 100 == 0:
                stats = sim.get_statistics()
                print(f"Step {step + 1}:")
                print(f"  Time: {stats['time']:.2f} s")
                print(f"  Avg Speed: {stats['average_speed']:.2f}")
                print(f"  Total KE: {stats['total_kinetic_energy']:.2f}")
                print(f"  Pressure: {stats['pressure']:.4f}")
                print(f"  Wall Collisions: {stats['wall_collisions']}")
                print(f"  Particle Collisions: {stats['particle_collisions']}")
                print()

        # Final statistics
        final_stats = sim.get_statistics()
        print("=" * 60)
        print("Final Statistics:")
        print("=" * 60)
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
