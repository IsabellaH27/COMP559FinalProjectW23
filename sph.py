import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm

BASE_DENSITY = 2
CONSTANT_FORCE = np.array([[0.0, -0.1]])
CONTAINERW = 60
CONTAINERH = 80
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFF = - 0.9
MAX_PARTICLES = 1000
P_MASS = 2
ISOTROPIC_EXPONENT = 30
SMOOTHING_LEN = 3
TIME_STEP_LENGTH = 0.01
NUM_TIMESTEPS = 2500
INSERT_GAP = 1
FIGURE_SIZE = (2, 3)
PLOT_EVERY = 1
SCATTER_DOT_SIZE = 2

CONTAINER_LIMIT_X = np.array([
    SMOOTHING_LEN,CONTAINERW - SMOOTHING_LEN,
])
CONTAINER_LIMIT_Y = np.array([
    SMOOTHING_LEN, CONTAINERH - SMOOTHING_LEN,
])

NORMALIZATION_DENSITY = (
    (315 * P_MASS) / (64 * np.pi * pow(SMOOTHING_LEN,9))
)
NORMALIZATION_PRESSURE_FORCE = (
    -(45 * P_MASS) / (np.pi * pow(SMOOTHING_LEN,6))
)
NORMALIZATION_VISCOUS_FORCE = (
    (45 * DYNAMIC_VISCOSITY * P_MASS) / (np.pi * pow(SMOOTHING_LEN,6))
)

def main():
    n_particles = 1

    pos = np.zeros((n_particles, 2))
    vel = np.zeros_like(pos)
    forces = np.zeros_like(pos)

    plt.figure(figsize=FIGURE_SIZE, dpi=160)

    for iter in tqdm(range(NUM_TIMESTEPS)):
        if iter % INSERT_GAP == 0 and n_particles < MAX_PARTICLES:
            new_positions = np.array([
                [10 + np.random.rand(), CONTAINER_LIMIT_Y[1]],
                [15 + np.random.rand(), CONTAINER_LIMIT_Y[1]],
                [20 + np.random.rand(), CONTAINER_LIMIT_Y[1]],
            ])
            
            new_velocities = np.array([
                [-3.0, -15.0],
                [-3.0, -15.0],
                [-3.0, -15.0],
            ])

            n_particles += 3

            pos = np.concatenate((pos, new_positions), axis=0)
            vel = np.concatenate((vel, new_velocities), axis=0)
        
        neighbor_ids, dist = neighbors.KDTree(
            pos,
        ).query_radius(
            pos,
            SMOOTHING_LEN,
            return_distance=True,
            sort_results=True,
        )

        dens = np.zeros(n_particles)

        for i in range(n_particles):
            for k, j in enumerate(neighbor_ids[i]):
                dens[i] += NORMALIZATION_DENSITY * pow((pow(SMOOTHING_LEN,2)-pow(dist[i][k],2 )),3)
        
        pressures = ISOTROPIC_EXPONENT * (dens - BASE_DENSITY)

        forces = np.zeros_like(pos)

        # Drop the element itself
        neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
        dist = [ np.delete(x, 0) for x in dist]

        for i in range(n_particles):
            for k, j in enumerate(neighbor_ids[i]):
                # Pressure force
                forces[i] += NORMALIZATION_PRESSURE_FORCE*(-(pos[j]-pos[i]) / dist[i][k]*(pressures[j]+pressures[i]) 
                    / (2*dens[j])*pow((SMOOTHING_LEN-dist[i][k]),2))
                # Viscous force
                forces[i] += NORMALIZATION_VISCOUS_FORCE * (
                    (vel[j]-vel[i]) / dens[j]*(SMOOTHING_LEN-dist[i][k])
                )
        
        # Force due to gravity
        forces += CONSTANT_FORCE * dens[:, np.newaxis]
        

        # Euler Step
        vel = vel + TIME_STEP_LENGTH * forces / dens[:, np.newaxis]
        pos = pos + TIME_STEP_LENGTH * vel

        # Enfore Boundary Conditions
        out_of_left_boundary = pos[:, 0] < CONTAINER_LIMIT_X[0]
        out_of_right_boundary = pos[:, 0] > CONTAINER_LIMIT_X[1]
        out_of_bottom_boundary = pos[:, 1] < CONTAINER_LIMIT_Y[0]
        out_of_top_boundary = pos[:, 1] > CONTAINER_LIMIT_Y[1]

        vel[out_of_left_boundary, 0]     *= DAMPING_COEFF
        pos [out_of_left_boundary, 0]      = CONTAINER_LIMIT_X[0]

        vel[out_of_right_boundary, 0]    *= DAMPING_COEFF
        pos [out_of_right_boundary, 0]     = CONTAINER_LIMIT_X[1]

        vel[out_of_bottom_boundary, 1]   *= DAMPING_COEFF
        pos [out_of_bottom_boundary, 1]    = CONTAINER_LIMIT_Y[0]

        vel[out_of_top_boundary, 1]      *= DAMPING_COEFF
        pos [out_of_top_boundary, 1]       = CONTAINER_LIMIT_Y[1]

        if iter % PLOT_EVERY == 0:
            plt.scatter(
                pos[:, 0],
                pos[:, 1],
                s=SCATTER_DOT_SIZE,
                c=pos[:, 1],
            )
            plt.ylim(CONTAINER_LIMIT_Y)
            plt.xlim(CONTAINER_LIMIT_X)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()


if __name__ == "__main__":
    main()
