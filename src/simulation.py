from src.environment import BallWorldInformation
from src.environment import BallSimulation
import numpy as np
import pygame
from src.sensor import Sensor
from src.particle_filter import ParticleFilter


class ParticleFilterSimulation(BallSimulation):
    """
    Real-time ball tracking simulation using particle filter estimation.
    
    Extends the basic ball simulation to include particle filter tracking with
    configurable sensor noise, observation dropout rates, and variable timing.
    Provides comprehensive visualization of true ball state, particle cloud,
    filter estimate, and sensor observations.
    """

    def __init__(
        self,
        world,
        initial_state,
        num_particles,
        sensor_noise,
        observation_rate=1.0,
        variable_dt=False,
    ):
        """
        Initialize particle filter simulation with tracking parameters.

        Sets up the simulation environment with a particle filter for state estimation.
        Configures sensor model, observation dropout simulation, and timing variations
        to test filter robustness under realistic conditions.

        Args:
            world (BallWorldInformation): Physics world configuration and boundaries
            initial_state (np.ndarray): True initial ball state [x, y, vx, vy] 
            num_particles (int): Number of particles for Monte Carlo estimation
            sensor_noise (float): Standard deviation of Gaussian observation noise
            observation_rate (float): Probability of receiving observation each frame [0,1]
            variable_dt (bool): Whether to use random time intervals between steps
        """
        super().__init__(world, initial_state)
        self.sensor = Sensor(noise_std=sensor_noise)
        self.particle_filter = ParticleFilter(
            num_particles, world, self.sensor, self.physics
        )
        self.observations = []
        self.show_particles = True
        self.observation_rate = observation_rate
        self.variable_dt = variable_dt
        self.steps_since_observation = 0
        self.total_observations = 0
        self.total_steps = 0

    def run(self, dt):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption(
            "Particle Filter Simulation - Press 'P' to toggle particles, 'SPACE' to pause"
        )
        clock = pygame.time.Clock()
        scale = 800 / max(self.world.width, self.world.height)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif (
                        event.key == pygame.K_p
                    ):  # Press 'P' to toggle particle visibility
                        self.show_particles = not self.show_particles
                        print(
                            f"Particles {'shown' if self.show_particles else 'hidden'}"
                        )

            if not self.paused:
                dt_actual = dt
                if self.variable_dt:
                    dt_actual = dt * np.random.uniform(0.5, 2.0)

                self.state = self.physics.simulate_collision(self.state, dt_actual)
                self.total_steps += 1

                got_observation = np.random.random() < self.observation_rate

                if got_observation:
                    true_position = self.state[:2]
                    observation = self.sensor.observe(true_position)
                    self.observations.append(observation)
                    self.particle_filter.update(observation)
                    self.particle_filter.resample()
                    self.steps_since_observation = 0
                    self.total_observations += 1
                else:
                    self.steps_since_observation += 1

                self.particle_filter.predict(dt_actual)

                _, y, vx, vy = self.state
                if (
                    abs(vx) < 1e-2
                    and abs(vy) < 1e-2
                    and abs(y - self.world.ball_radius) < 1e-3
                ):
                    print(
                    )
                    self.running = False

            self.screen.fill((255, 255, 255))

            x, y, _, _ = self.state
            screen_x = int(x * scale)
            screen_y = 800 - int(y * scale)
            ball_radius_screen = int(self.world.ball_radius * scale)
            pygame.draw.circle(
                self.screen,
                (0, 0, 255),
                (screen_x, screen_y),
                ball_radius_screen,
            )

            if self.show_particles:
                for particle in self.particle_filter.particles:
                    px, py = particle[:2]
                    screen_px = int(px * scale)
                    screen_py = 800 - int(py * scale)
                    pygame.draw.circle(
                        self.screen, (255, 0, 0), (screen_px, screen_py), 2
                    )

            estimate_pos = self.particle_filter.get_position_estimate()
            screen_est_x = int(estimate_pos[0] * scale)
            screen_est_y = 800 - int(estimate_pos[1] * scale)
            pygame.draw.circle(
                self.screen,
                (255, 255, 0),
                (screen_est_x, screen_est_y),
                ball_radius_screen,
            )

            if self.observations:
                obs_x, obs_y = self.observations[-1]
                screen_obs_x = int(obs_x * scale)
                screen_obs_y = 800 - int(obs_y * scale)
                pygame.draw.circle(
                    self.screen, (0, 255, 0), (screen_obs_x, screen_obs_y), 5
                )

            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, 800, 800), 5)

            # Add status text with comprehensive information
            font = pygame.font.Font(None, 24)
            y_offset = 10

            # Particle visibility status
            status_text = f"Particles: {'ON' if self.show_particles else 'OFF'} (Press P to toggle)"
            text_surface = font.render(status_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

            if self.total_steps > 0:
                obs_rate = 100 * self.total_observations / self.total_steps
                obs_text = f"Observations: {self.total_observations}/{self.total_steps} ({obs_rate:.1f}%)"
                text_surface = font.render(obs_text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 25

            if self.steps_since_observation > 0:
                dropout_text = (
                    f"Steps since observation: {self.steps_since_observation}"
                )
                text_surface = font.render(dropout_text, True, (255, 0, 0))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 25

            config_text = f"Config: {len(self.particle_filter.particles)} particles, noise={self.sensor.noise_std:.1f}, obs_rate={self.observation_rate:.1f}"
            text_surface = font.render(config_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    """
    Execute the particle filter demonstration with configured parameters.
    
    Creates a ball physics world and runs a particle filter simulation with
    parameters optimized for demonstrating tracking capabilities under
    uncertainty. Includes sensor noise, observation dropouts, and variable
    timing to test filter robustness.
    """
    world = BallWorldInformation(
        width=50,
        height=50,
        gravity=9.8,
        ball_radius=2,
        bounce_discount=0.8,
        air_discount=0.995,
        ground_discount=0.7,
    )

    initial_state = np.array([25.0, 40.0, 15.0, 5.0])

    sim = ParticleFilterSimulation(
        world,
        initial_state,
        num_particles=1000,
        sensor_noise=2.0,
        observation_rate=0.1,
        variable_dt=True,
    )

    print("Particle Filter Ball Tracking Simulation")
    print("========================================")
    print("Controls:")
    print("  SPACE: Pause/unpause simulation")
    print("  P: Toggle particle visibility")
    print("  ESC/Close: Exit simulation")
    print("")
    print("Visualization:")
    print("  Blue circle: True ball position")
    print("  Red dots: Particle cloud (when visible)")
    print("  Yellow circle: Particle filter estimate")
    print("  Green dot: Latest noisy observation")
    print("")

    sim.run(dt=0.01)


if __name__ == "__main__":
    main()
