"""
Ball physics simulation with precise collision detection and realistic bouncing.
"""

import numpy as np
import pygame


class BallWorldInformation:
    """Configuration parameters for the ball physics world."""

    def __init__(
        self,
        width,
        height,
        gravity,
        ball_radius,
        bounce_discount,
        air_discount,
        ground_discount,
    ):
        """
        Initialize world parameters.

        Args:
            width (float): World width in units
            height (float): World height in units
            gravity (float): Gravitational acceleration (positive downward)
            ball_radius (float): Ball radius in units
            bounce_discount (float): Energy loss factor on bounce (0-1)
            air_discount (float): Air resistance factor per time unit (0-1)
            ground_discount (float): Ground friction factor (0-1)
        """
        self.width = width
        self.height = height
        self.gravity = gravity
        self.ball_radius = ball_radius
        self.bounce_discount = bounce_discount
        self.air_discount = air_discount
        self.ground_discount = ground_discount
        self.bounce_stop_threshold = 1e-10


class BallPhysics:
    """Physics engine with precise collision detection."""

    def __init__(self, world: BallWorldInformation):
        """
        Initialize physics engine.

        Args:
            world (BallWorldInformation): World configuration parameters
        """
        self.world = world

    def simulate_collision(self, state, dt):
        """
        Simulate ball motion with collision detection and response.

        Args:
            state (np.ndarray): Current state [x, y, vx, vy]
            dt (float): Time step for simulation

        Returns:
            np.ndarray: Updated state [x, y, vx, vy]
        """
        x, y, vx, vy = state
        time_remaining = dt

        if x < self.world.ball_radius:
            x = self.world.ball_radius
            vx = abs(vx)
        elif x > self.world.width - self.world.ball_radius:
            x = self.world.width - self.world.ball_radius
            vx = -abs(vx)

        if y < self.world.ball_radius:
            y = self.world.ball_radius
            vy = max(0, vy)
        elif y > self.world.height - self.world.ball_radius:
            y = self.world.height - self.world.ball_radius
            vy = min(0, vy)

        while time_remaining > 1e-6:
            tx = float("inf")
            ty = float("inf")

            if vx > 0:
                tx = (self.world.width - self.world.ball_radius - x) / vx
            elif vx < 0:
                tx = (self.world.ball_radius - x) / vx

            if vy > 0:
                ty = (self.world.height - self.world.ball_radius - y) / vy
            elif vy < 0:
                ty = (y - self.world.ball_radius) / (-vy)

            t_collision = min(tx, ty)

            if t_collision > time_remaining or t_collision < 0:
                x += vx * time_remaining
                y += vy * time_remaining - 0.5 * self.world.gravity * time_remaining**2
                vy -= self.world.gravity * time_remaining
                break

            x += vx * t_collision
            y += vy * t_collision - 0.5 * self.world.gravity * t_collision**2
            vy -= self.world.gravity * t_collision

            if t_collision == tx:
                vx = -vx * self.world.bounce_discount
            if t_collision == ty:
                if vy < 0:
                    y = self.world.ball_radius
                    new_vy = -vy * self.world.bounce_discount
                    if abs(new_vy) < self.world.bounce_stop_threshold:
                        vy = 0
                    else:
                        vy = new_vy
                elif vy > 0:
                    y = self.world.height - self.world.ball_radius
                    vy = -vy * self.world.bounce_discount

            vx *= self.world.air_discount**t_collision
            vy *= self.world.air_discount**t_collision

            time_remaining -= t_collision

        if abs(y - self.world.ball_radius) < 1e-3:
            y = self.world.ball_radius
            vy = 0
            vx *= self.world.ground_discount**dt
            if abs(vx) < 1e-2:
                vx = 0

        return np.array([x, y, vx, vy])


class BallSimulation:
    """Real-time visualization of ball physics simulation using PyGame."""

    def __init__(self, world: BallWorldInformation, initial_state):
        """
        Initialize simulation.

        Args:
            world (BallWorldInformation): World configuration parameters
            initial_state (np.ndarray): Initial ball state [x, y, vx, vy]
        """
        self.world = world
        self.state = initial_state
        self.physics = BallPhysics(world)
        self.running = True
        self.paused = False
        self.screen = None

    def run(self, dt):
        """
        Run the simulation with real-time visualization.

        Args:
            dt (float): Time step for physics simulation
        """
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Ball Physics Simulation")
        clock = pygame.time.Clock()

        scale = 800 / max(self.world.width, self.world.height)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused

            if not self.paused:
                self.state = self.physics.simulate_collision(self.state, dt)

                x, y, vx, vy = self.state
                if (
                    abs(vx) < 1e-2
                    and abs(vy) < 1e-2
                    and abs(y - self.world.ball_radius) < 1e-3
                ):
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

            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, 800, 800), 5)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    """Main function to run the ball physics simulation."""
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
    sim = BallSimulation(world, initial_state)
    sim.run(dt=0.01)


if __name__ == "__main__":
    main()
