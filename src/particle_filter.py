import numpy as np
from scipy.stats import norm


class ParticleFilter:
    """
    Particle filter for tracking ball position and velocity using Monte Carlo methods.
    
    This implementation maintains a population of particles (hypotheses) about the ball's
    state and uses observation likelihood to weight and resample particles over time.
    Each particle represents a complete ball state [x, y, vx, vy].
    """

    def __init__(self, num_particles, world, sensor, physics):
        """
        Initialize the particle filter with random particle distribution.

        Particles are initialized uniformly across the world space with wide velocity
        ranges to account for unknown initial ball conditions. Process noise parameters
        are set to maintain particle diversity during tracking.

        Args:
            num_particles (int): Number of particles to maintain in the filter
            world (BallWorldInformation): World configuration with boundaries and physics
            sensor (Sensor): Sensor model for observation likelihood calculation
            physics (BallPhysics): Physics engine for particle state transitions
        """
        self.num_particles = num_particles
        self.world = world
        self.sensor = sensor
        self.physics = physics

        self.particles = np.random.uniform(
            [world.ball_radius, world.ball_radius, -30, -30],
            [world.width - world.ball_radius, world.height - world.ball_radius, 30, 30],
            size=(num_particles, 4),
        )
        self.weights = np.ones(num_particles) / num_particles

        self.process_noise_pos = 0.1
        self.process_noise_vel = 0.5

    def predict(self, dt):
        """
        Predict the next state of all particles using the physics model.
        
        Applies the same physics simulation used for the true ball to each particle,
        then adds process noise to maintain particle diversity and prevent filter
        degeneracy. Particles are constrained to remain within world boundaries.
        
        Args:
            dt (float): Time step for physics simulation
        """
        for i in range(self.num_particles):
            self.particles[i] = self.physics.simulate_collision(self.particles[i], dt)

            pos_noise = np.random.normal(0, self.process_noise_pos, 2)
            vel_noise = np.random.normal(0, self.process_noise_vel, 2)

            self.particles[i, :2] += pos_noise
            self.particles[i, 2:] += vel_noise

            self.particles[i, 0] = np.clip(
                self.particles[i, 0],
                self.world.ball_radius,
                self.world.width - self.world.ball_radius,
            )
            self.particles[i, 1] = np.clip(
                self.particles[i, 1],
                self.world.ball_radius,
                self.world.height - self.world.ball_radius,
            )

    def update(self, observation):
        """
        Update particle weights based on observation likelihood.
        
        Calculates how likely each particle's position is given the noisy observation
        using Gaussian probability density functions. Particles closer to the observation
        receive higher weights. Handles edge cases where all particles have zero likelihood
        by resetting to uniform distribution.

        Args:
            observation (np.ndarray): Noisy sensor observation of ball position [x, y]
        """
        for i in range(self.num_particles):
            particle_pos = self.particles[i, :2]

            likelihood_x = norm.pdf(
                observation[0], particle_pos[0], self.sensor.noise_std
            )
            likelihood_y = norm.pdf(
                observation[1], particle_pos[1], self.sensor.noise_std
            )

            self.weights[i] = likelihood_x * likelihood_y

        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        """
        Resample particles based on their weights using systematic resampling.
        
        Evaluates the effective sample size to determine if resampling is necessary.
        When most particles have negligible weights, performs multinomial resampling
        where particles with higher weights are more likely to be selected and
        duplicated. Resets all weights to uniform after resampling.
        """
        effective_n = 1.0 / np.sum(self.weights**2)

        if effective_n < self.num_particles / 2:
            indices = np.random.choice(
                range(self.num_particles),
                size=self.num_particles,
                replace=True,
                p=self.weights,
            )
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """
        Compute the weighted average state estimate from all particles.
        
        Returns the expectation of the posterior distribution represented by
        the weighted particle set. This provides the best point estimate
        of the ball's current position and velocity.
        
        Returns:
            np.ndarray: Estimated state [x, y, vx, vy] as weighted particle average
        """
        return np.average(self.particles, axis=0, weights=self.weights)

    def get_position_estimate(self):
        """
        Extract only the position component of the state estimate.
        
        Convenience method that returns just the x,y coordinates from the
        full state estimate, useful for visualization and tracking metrics.
        
        Returns:
            np.ndarray: Estimated position [x, y] from weighted particle average
        """
        full_estimate = self.estimate()
        return full_estimate[:2]
