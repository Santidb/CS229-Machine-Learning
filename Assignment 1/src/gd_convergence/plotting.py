import argparse
import numpy as np
import matplotlib.pyplot as plt
from experiment import J, update_theta, gradient_descend, A, theta_0

angle = np.pi/4
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])
A_rotated = R.dot(A).dot(R.T)

def trajectory(theta_0, lr, epsilon=1e-50):
    traj = [theta_0]
    def update_theta_trajectory(theta, lr):
        new_theta = update_theta(theta, lr)
        traj.append(new_theta)
        return new_theta
    gradient_descend(J, theta_0, lr, update_theta_trajectory)
    return np.vstack(traj)

def trajectory_rotated(theta_0, lr, epsilon=1e-50):
    def J_rotated(theta):
        return theta.T.dot(A_rotated).dot(theta)
    traj = [theta_0]
    def update_theta_trajectory(theta, lr):
        new_theta = theta - lr * (A_rotated + A_rotated.T).dot(theta)
        traj.append(new_theta)
        return new_theta
    gradient_descend(J_rotated, theta_0, lr, update_theta_trajectory)
    return np.vstack(traj)


def plot_trajectory(trajectories, lrs):
    num_points = 50
    xx, yy = np.meshgrid(np.linspace(-2, 2, num_points), np.linspace(-2, 2, num_points))
    thetas = np.stack((xx, yy), axis=2).reshape(-1, 2)
    Js = np.sum(thetas.dot(A) * thetas, axis=-1).reshape(-1, num_points)
    plt.figure()
    plt.contour(xx, yy, Js, 10)
    for t, _ in zip(trajectories, lrs):
        x, y = zip(*t)
        plt.plot(x, y)
    plt.legend(lrs)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.title('Original A Trajectory')
    plt.savefig('trajectories.png')

def plot_trajectory_rotated(trajectories, lrs):
    num_points = 50
    xx, yy = np.meshgrid(np.linspace(-2, 2, num_points), np.linspace(-2, 2, num_points))
    thetas = np.stack((xx, yy), axis=2).reshape(-1, 2)
    Js = np.sum(thetas.dot(A_rotated) * thetas, axis=-1).reshape(-1, num_points)
    plt.figure()
    plt.contour(xx, yy, Js, 15)
    for t, _ in zip(trajectories, lrs):
        x, y = zip(*t)
        plt.plot(x, y)
    plt.legend(lrs)
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.title('Rotated A Trajectory')
    plt.savefig('trajectories_rotated.png')

def print_num_iterations(trajectories, lrs, prefix):
    for t, lr in zip(trajectories, lrs):
        converged = 'converged' if J(t[-1]) < 1e-25 else 'did not converge'
        print(f"{prefix}, learning rate {lr}] GD took {len(t)} iterations and {converged}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot the trajectories given learning rates')
    parser.add_argument('lrs', nargs='+', help='a list of learning rates')
    args = parser.parse_args()
    lrs = list(map(float, args.lrs))
    ts = []
    for lr in lrs:
        ts.append(trajectory(theta_0, lr))
    plot_trajectory(ts, lrs)
    print_num_iterations(ts, lrs, '[diagonal A')
    ts = []
    for lr in lrs:
        ts.append(trajectory_rotated(theta_0, lr))
    plot_trajectory_rotated(ts, lrs)
    print_num_iterations(ts, lrs, '[rotated A')
