# Particle Filter Ball Tracking

This project implements a particle filter to track a bouncing ball with noisy position observations, meeting the requirements for assignment P2.1.

## Project Overview

The system simulates a ball bouncing in a 2D environment with realistic physics (gravity, collisions, friction) and uses a particle filter to estimate the ball's position and velocity from noisy observations. The initial position and velocity of the ball are unknown to the estimation system.

## Features

### Core Requirements Met:
- ✅ **Particle filter implementation** for state estimation of position and velocity
- ✅ **Unknown initial conditions** - particles initialized over large range (50×50 meters)
- ✅ **Configurable sensor uncertainty** - parameterizable observation noise
- ✅ **Variable observation intervals** - time spans between observations can vary
- ✅ **Observation dropouts** - sensor can fail completely for periods
- ✅ **State estimation during sensor failure** - filter continues to predict during dropouts
- ✅ **Real-time visualization** with comprehensive information display

### Technical Implementation:
- **Physics simulation**: Realistic ball dynamics with gravity, bouncing, air resistance, and friction
- **Sensor model**: Configurable Gaussian noise on position measurements
- **Particle filter**: Standard predict-update-resample cycle with adaptive resampling
- **Process noise**: Added during prediction to prevent particle depletion
- **Boundary handling**: Particles constrained within valid world bounds

## File Structure

```
src/
├── environment.py          # Ball physics and world simulation
├── particle_filter.py      # Particle filter implementation
├── sensor.py              # Sensor model with configurable noise
├── simulation.py          # Main simulation with visualization
└── perfect_environment.py # Clean physics simulation (reference)

pyproject.toml             # Project dependencies
README.md                  # This file
```

## Requirements

- Python 3.11+
- Dependencies managed via pyproject.toml:
  - numpy (numerical computations)
  - pygame (visualization)
  - scipy (statistical functions)

## Installation

```bash
# Clone or download the project
cd Reasoning-and-Decision-Making-under-Uncertainty

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install numpy pygame scipy
```

## Running the Simulation

```bash
python src/simulation.py
```

## Controls

- **SPACE**: Pause/unpause simulation
- **P**: Toggle particle visibility (show/hide red dots)
- **ESC/Close window**: Exit simulation

## Visualization

The simulation displays multiple elements:

- **Blue circle**: True ball position (ground truth)
- **Red dots**: Individual particles (when visible) - each represents a hypothesis about ball state
- **Yellow circle**: Particle filter estimate (weighted average of particles)
- **Green dot**: Latest noisy sensor observation
- **Status display**: Shows observation statistics, sensor failure periods, and configuration

## Configuration Parameters

Key parameters can be modified in `main()` function in `simulation.py`:

```python
# Core particle filter parameters
num_particles=1000      # Number of particles (1000+ recommended for unknown initial position)
sensor_noise=2.0        # Standard deviation of position measurement noise
observation_rate=0.8    # Probability of receiving observation each frame (0.8 = 20% dropout)
variable_dt=True        # Enable variable time intervals between observations

# World parameters
width=50, height=50     # World dimensions (ball starts somewhere in this range)
gravity=9.8            # Gravitational acceleration
ball_radius=2          # Ball size
bounce_discount=0.8    # Energy loss on bounce
```

## Algorithm Details

### Particle Filter Implementation

The particle filter follows the standard three-step cycle:

1. **Predict Step** (`predict(dt)`):
   - Apply physics model to each particle using realistic ball dynamics
   - Add process noise to prevent particle depletion
   - Handle variable time intervals when enabled
   - Ensure particles remain within world boundaries

2. **Update Step** (`update(observation)`):
   - Calculate likelihood of each particle given the observation
   - Weight particles based on Gaussian likelihood function
   - Handle cases where all weights become zero

3. **Resample Step** (`resample()`):
   - Check effective sample size to determine if resampling is needed
   - Use weighted resampling to redistribute particles
   - Only resample when effective sample size drops below threshold

### State Representation

Each particle represents a complete ball state:
```
[x, y, vx, vy]
```
- `x, y`: Position coordinates
- `vx, vy`: Velocity components

### Handling Sensor Failures

The system robustly handles observation dropouts:
- Filter continues prediction steps even without observations
- Status display shows periods of sensor failure in red
- Particles spread out during failure periods due to process noise
- Filter reconverges when observations resume

## Performance Analysis

The simulation provides real-time feedback on filter performance:

- **Observation rate**: Percentage of time steps with valid observations
- **Steps since observation**: Counter showing current sensor failure duration
- **Particle count**: Number of hypotheses being tracked
- **Configuration display**: Key parameters for current run

## Assignment Requirements Compliance

This implementation specifically addresses all P2.1 requirements:

### ✅ Single Ball Tracking (n=1)
- Estimates position and velocity of one ball from noisy observations
- Unknown initial launch position and direction
- Large initial uncertainty handled by 1000+ particles spread over 50×50m area

### ✅ Configurable Uncertainty
- `sensor_noise` parameter controls observation uncertainty
- Process noise parameters prevent particle depletion
- Boundary constraints ensure realistic particle states

### ✅ Variable Observation Intervals
- `variable_dt=True` enables random time intervals between predictions
- Time steps vary randomly between 0.5× and 2× base rate
- Simulates realistic sensor timing variations

### ✅ Observation Dropouts
- `observation_rate` parameter controls probability of sensor failure
- Filter continues state estimation during dropout periods
- Visual feedback shows sensor failure periods

### ✅ Unknown Initial Conditions
- Particles initialized uniformly over entire world space
- Velocity initialization covers wide range [-20, 20] m/s
- No prior knowledge assumed about launch parameters

## Extension to Multiple Balls

While this implementation focuses on single ball tracking as required, the architecture supports extension to multiple balls (n>1) by:

1. **State augmentation**: Extend state vector to `[x1,y1,vx1,vy1, x2,y2,vx2,vy2, ...]`
2. **Data association**: Implement nearest neighbor or probabilistic association
3. **Multi-modal estimation**: Use clustering algorithms to extract multiple ball positions

## Testing and Validation

To validate the filter performance:

1. **Run with different noise levels**: Try `sensor_noise` values from 0.5 to 5.0
2. **Test observation dropout**: Vary `observation_rate` from 0.3 to 1.0
3. **Unknown initial conditions**: Modify initial ball position and velocity
4. **Toggle particle visibility**: Compare estimate (yellow) to ground truth (blue)

## Notes for Interview Discussion

Key topics for interview presentation:

1. **Algorithm explanation**: How particle filter predict-update-resample cycle works
2. **Parameter sensitivity**: Effect of particle count, noise levels, observation rates
3. **Failure handling**: How system maintains estimates during sensor dropouts
4. **Performance metrics**: Tracking accuracy, computational efficiency
5. **Design decisions**: Why certain parameters and approaches were chosen

## Author

Implementation for Assignment P2.1 - Particle Filter for Ball Tracking
Course: Reasoning and Decision Making under Uncertainty