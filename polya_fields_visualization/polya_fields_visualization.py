import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from typing import Callable
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Union
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .utils import validate_input, conjugate_function


def visualization(f: Callable, x_range: Union[tuple[float, float, int], list[float]],
                  y_range: Union[tuple[float, float, int], list[float]], **kwargs):
    x, y, map_f = validate_input(f, x_range, y_range, "2d", True, **kwargs)
    # build vectors
    eps = map_f["eps"]
    X, Y = np.meshgrid(x, y)
    Z = conjugate_function(f, x, y, eps)
    U = np.real(Z)
    V = np.imag(Z)
    # normalization
    normalization = np.sqrt(U ** 2 + V ** 2) + eps
    U_norm = U / normalization
    V_norm = V / normalization
    plt.figure(figsize=map_f["fig_size"])
    if map_f["type_plot"] == "vector":
        plt.quiver(X, Y, U_norm, V_norm, normalization,cmap=map_f["color_vector"], pivot='mid')
        plt.colorbar(label='Vector length')
    elif map_f["type_plot"] == "stream":
        strm = plt.streamplot(X, Y, U_norm, V_norm, color=U, linewidth=map_f["width_line"], cmap=map_f["color_vector"])
        plt.colorbar(strm.lines)
    if map_f["contour_func"] is not None:
        theta = np.linspace(0, 2*np.pi, 1000)
        z_contour = map_f["contour_func"](theta)
        x_contour = np.real(z_contour)
        y_contour = np.imag(z_contour)
        
        x_contour = np.append(x_contour, x_contour[0])
        y_contour = np.append(y_contour, y_contour[0])
        
        plt.plot(x_contour, y_contour, 
                color=map_f['contour_color'],
                linewidth=map_f['contour_linewidth'])
    plt.xlabel(map_f["lable_x"])
    plt.ylabel(map_f["lable_y"])
    plt.title(map_f["title_plot"])
    plt.grid()
    plt.axhline(0, color=map_f["color_line_x"], linewidth=map_f["width_line_x"])
    plt.axvline(0, color=map_f["color_line_y"], linewidth=map_f["width_line_y"])
    plt.gca().set_aspect('equal')
    plt.show()


def visualization_anim(f: Callable, x_range: Union[tuple[float, float, int], list[float]],
                  y_range: Union[tuple[float, float, int], list[float]], **kwargs):
    x, y, map_f = validate_input(f, x_range, y_range, "2d", False, **kwargs)
    X, Y = np.meshgrid(x, y)
    eps = map_f["eps"]
    Z = conjugate_function(f, x, y, eps)
    U = np.real(Z)
    V = np.imag(Z)

    normalization = np.sqrt(U ** 2 + V ** 2) + eps
    U_norm = U / normalization
    V_norm = V / normalization

    fig = plt.figure(figsize=map_f["fig_size"])
    ax = fig.add_subplot()
    if map_f["show_vectors"]:
        if map_f["type_plot"] == "vector":
            q = ax.quiver(X, Y, U_norm, V_norm, normalization, cmap=map_f["color_vector"], pivot='mid')
            plt.colorbar(q, label='Vector length')
        elif map_f["type_plot"] == "stream":
            strm = ax.streamplot(X, Y, U_norm, V_norm, color=U, linewidth=2, cmap=map_f["color_vector"])
            plt.colorbar(strm.lines)

    if map_f["contour_func"] is not None:
        theta = np.linspace(0, 2*np.pi, 1000)
        z_contour = map_f["contour_func"](theta)
        x_contour = np.real(z_contour)
        y_contour = np.imag(z_contour)
        
        x_contour = np.append(x_contour, x_contour[0])
        y_contour = np.append(y_contour, y_contour[0])
        
        plt.plot(x_contour, y_contour, 
                 color=map_f['contour_color'],
                linewidth=map_f['contour_linewidth'])

    ax.set_xlabel(map_f["lable_x"])
    ax.set_ylabel(map_f["lable_y"])
    ax.set_title(map_f["title_plot"])
    ax.grid()
    ax.axhline(0, color=map_f["color_line_x"], linewidth=map_f["width_line_x"])
    ax.axvline(0, color=map_f["color_line_y"], linewidth=map_f["width_line_y"])

    interpolator_u = RegularGridInterpolator((x, y), U_norm.T, bounds_error=False, fill_value=0)
    interpolator_v = RegularGridInterpolator((x, y), V_norm.T, bounds_error=False, fill_value=0)

    x_min, x_max = x.min(), y.max()
    y_min, y_max = x.min(), y.max()
    particles = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(map_f["num_particles"], 2))
    trails = np.full((map_f["num_particles"], map_f["trail_length"], 2), np.nan)

    position_history = [[] for _ in range(map_f["num_particles"])]

    scat = ax.scatter(particles[:, 0], particles[:, 1], color=map_f["color_particles"], s=5)
    trail_lines = LineCollection([], color=map_f["color_particles"], linewidths=map_f["trail_width"])
    ax.add_collection(trail_lines)

    def update(frame):
        nonlocal particles, trails, position_history
        for i in range(map_f["num_particles"]):
            if len(position_history[i]) >= 2:
                recent_pos = np.array(position_history[i][-2:])
                if np.all(np.linalg.norm(recent_pos - recent_pos[0], axis=1) < 0.001):
                    particles[i] = np.random.uniform([x_min, y_min], [x_max, y_max])
                    trails[i] = np.full((map_f["trail_length"], 2), np.nan)
                    position_history[i] = []
                    continue
        
            x, y = particles[i]
            u = interpolator_u([[x, y]])[0]
            v = interpolator_v([[x, y]])[0]
            particles[i, 0] += u * map_f["dt"]
            particles[i, 1] += v * map_f["dt"]

            position_history[i].append(particles[i].copy())
            if len(position_history[i]) > 2:
                position_history[i].pop(0)

            if not (x_min <= particles[i, 0] <= x_max and y_min <= particles[i, 1] <= y_max):
                particles[i] = np.random.uniform([x_min, y_min], [x_max, y_max])
                trails[i] = np.full((map_f["trail_length"], 2), np.nan)
                position_history[i] = []
                continue

            trails[i] = np.roll(trails[i], shift=-1, axis=0)
            trails[i][-1] = particles[i]

        scat.set_offsets(particles)
        trail_lines.set_segments(trails)
        return scat, trail_lines

    ani = animation.FuncAnimation(fig, update, frames=map_f["frames"], interval=map_f["interval"], blit=True)
    plt.gca().set_aspect('equal')
    plt.show()




def visualization_sphere(f: Callable,
                         x_range: Union[tuple[float, float, int], list[float]],
                         y_range: Union[tuple[float, float, int], list[float]],
                         **kwargs):

    x, y, map_f = validate_input(f, x_range, y_range, "3d", True, **kwargs)
    X, Y = np.meshgrid(x, y)
    eps = map_f["eps"]
    Z = np.array(conjugate_function(f, x, y, eps), dtype=complex) 
    U = Z.real
    V = Z.imag

    r_squared = X ** 2 + Y ** 2
    denominator = 1 + r_squared
    xi = 2 * X / denominator
    eta = 2 * Y / denominator
    zeta = (r_squared - 1) / denominator

    dxi_dX = 2 * (1 - X ** 2 + Y ** 2) / denominator ** 2
    dxi_dY = -4 * X * Y / denominator ** 2
    deta_dX = -4 * X * Y / denominator ** 2
    deta_dY = 2 * (1 + X ** 2 - Y ** 2) / denominator ** 2
    dzeta_dX = 4 * X / denominator ** 2
    dzeta_dY = 4 * Y / denominator ** 2


    U_3d = dxi_dX * U + dxi_dY * V
    V_3d = deta_dX * U + deta_dY * V
    W_3d = dzeta_dX * U + dzeta_dY * V

    normalization = np.sqrt(U_3d ** 2 + V_3d ** 2 + W_3d ** 2) + eps
    U_3d_norm = U_3d / normalization
    V_3d_norm = V_3d / normalization
    W_3d_norm = W_3d / normalization

    colors = normalization / np.max(normalization)
    colors_rgba = plt.get_cmap(map_f["color_vector"])(colors.ravel())

    fig = plt.figure(figsize=map_f['fig_size'])
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sp = np.outer(np.cos(u), np.sin(v))
    y_sp = np.outer(np.sin(u), np.sin(v))
    z_sp = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sp, y_sp, z_sp, color='b', alpha=0.1)

    ax.quiver(xi.ravel(), eta.ravel(), zeta.ravel(),
              U_3d_norm.ravel(), V_3d_norm.ravel(), W_3d_norm.ravel(),
              colors=colors_rgba,
              length=map_f["vector_length"], arrow_length_ratio=0.3, normalize=False)
    if map_f["contour_func"] != None:
        contour_func = map_f["contour_func"]
        t = np.linspace(0, 1, 1000)
        z_contour = contour_func(t)
        
        X_contour = np.real(z_contour)
        Y_contour = np.imag(z_contour)
        R_sq = X_contour**2 + Y_contour**2
        denom = 1 + R_sq
        xi_contour = 2 * X_contour / denom
        eta_contour = 2 * Y_contour / denom
        zeta_contour = (R_sq - 1) / denom
        
        xi_contour = np.append(xi_contour, xi_contour[0])
        eta_contour = np.append(eta_contour, eta_contour[0])
        zeta_contour = np.append(zeta_contour, zeta_contour[0])
        
        ax.plot(xi_contour, eta_contour, zeta_contour,
                color=map_f['contour_color'],
                linewidth=map_f['contour_linewidth'])

    ax.set_xlabel(map_f["lable_x"])
    ax.set_ylabel(map_f["lable_y"])
    ax.set_zlabel(map_f["lable_z"])
    ax.set_title(map_f["title_plot"])
    ax.xaxis.line.set_color(map_f["color_line_x"])
    ax.yaxis.line.set_color(map_f["color_line_y"])
    ax.zaxis.line.set_color(map_f["color_line_z"])

    ax.xaxis.line.set_linewidth(map_f["width_line_x"])
    ax.yaxis.line.set_linewidth(map_f["width_line_y"])
    ax.zaxis.line.set_linewidth(map_f["width_line_z"])
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.show()

def animate_sphere(f: Callable,
                         x_range: Union[tuple[float, float, int], list[float]],
                         y_range: Union[tuple[float, float, int], list[float]],
                         **kwargs):

    x, y, map_f = validate_input(f, x_range, y_range, "3d", False, **kwargs)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    eps = map_f["eps"]
    X, Y = np.meshgrid(x, y)
    Z = np.array(conjugate_function(f, x, y, eps), dtype=complex)
    U = Z.real
    V = Z.imag

    r_squared = X ** 2 + Y ** 2
    denominator = 1 + r_squared
    xi = 2 * X / denominator
    eta = 2 * Y / denominator
    zeta = (r_squared - 1) / denominator

    dxi_dX = 2 * (1 - X ** 2 + Y ** 2) / denominator ** 2
    dxi_dY = -4 * X * Y / denominator ** 2
    deta_dX = -4 * X * Y / denominator ** 2
    deta_dY = 2 * (1 + X ** 2 - Y ** 2) / denominator ** 2
    dzeta_dX = 4 * X / denominator ** 2
    dzeta_dY = 4 * Y / denominator ** 2

    U_3d = dxi_dX * U + dxi_dY * V
    V_3d = deta_dX * U + deta_dY * V
    W_3d = dzeta_dX * U + dzeta_dY * V

    normalization = np.sqrt(U_3d ** 2 + V_3d ** 2 + W_3d ** 2) + eps
    U_3d_norm = U_3d / normalization
    V_3d_norm = V_3d / normalization
    W_3d_norm = W_3d / normalization


    colors = normalization / np.max(normalization)
    colors_rgba = plt.get_cmap(map_f["color_vector"])(colors.ravel())


    interpolator_u = RegularGridInterpolator((x, y), U_3d_norm.T, bounds_error=False, fill_value=0)
    interpolator_v = RegularGridInterpolator((x, y), V_3d_norm.T, bounds_error=False, fill_value=0)
    interpolator_w = RegularGridInterpolator((x, y), W_3d_norm.T, bounds_error=False, fill_value=0)

    fig = plt.figure(figsize=map_f["fig_size"])
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sp = np.outer(np.cos(u), np.sin(v))
    y_sp = np.outer(np.sin(u), np.sin(v))
    z_sp = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sp, y_sp, z_sp, color='b', alpha=0.1)

    if map_f["show_vectors"]:
        ax.quiver(xi.ravel(), eta.ravel(), zeta.ravel(),
                  U_3d_norm.ravel(), V_3d_norm.ravel(), W_3d_norm.ravel(),
                  color=colors_rgba, cmap='cool', length=map_f["vector_length"], normalize=False,
                  alpha=0.6, arrow_length_ratio=0.3)
    if map_f["contour_func"] != None:
        contour_func = map_f["contour_func"]
        t = np.linspace(0, 1, 1000)
        z_contour = contour_func(t)
        
        X_contour = np.real(z_contour)
        Y_contour = np.imag(z_contour)
        R_sq = X_contour**2 + Y_contour**2
        denom = 1 + R_sq
        xi_contour = 2 * X_contour / denom
        eta_contour = 2 * Y_contour / denom
        zeta_contour = (R_sq - 1) / denom
        
        xi_contour = np.append(xi_contour, xi_contour[0])
        eta_contour = np.append(eta_contour, eta_contour[0])
        zeta_contour = np.append(zeta_contour, zeta_contour[0])
        
        ax.plot(xi_contour, eta_contour, zeta_contour,
                color=map_f['contour_color'],
                linewidth=map_f['contour_linewidth'])
        ax.set_xlabel(map_f["lable_x"])

    ax.set_ylabel(map_f["lable_y"])
    ax.set_zlabel(map_f["lable_z"])
    ax.set_title(map_f["title_plot"])
    ax.xaxis.line.set_color(map_f["color_line_x"])
    ax.yaxis.line.set_color(map_f["color_line_y"])
    ax.zaxis.line.set_color(map_f["color_line_z"])

    ax.xaxis.line.set_linewidth(map_f["width_line_x"])
    ax.yaxis.line.set_linewidth(map_f["width_line_y"])
    ax.zaxis.line.set_linewidth(map_f["width_line_z"])

    particles = np.random.uniform(low=[-1, -1, -1], high=[1, 1, 1], size=(map_f["num_particles"], 3))
    particles /= np.linalg.norm(particles, axis=1, keepdims=True)
    trails = np.full((map_f["num_particles"], map_f["trail_length"], 3), np.nan)

    position_history = [[] for _ in range(map_f["num_particles"])]

    scat = ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], color=map_f["color_particles"], s=10)
    trail_lines = Line3DCollection([], color=map_f["color_particles"], linewidths=map_f["trail_width"])
    ax.add_collection(trail_lines)

    def update(frame):
        nonlocal particles, trails, position_history
        for i in range(map_f["num_particles"]):
            pos = particles[i]
            x_proj = pos[0] / (1 - pos[2] + eps)
            y_proj = pos[1] / (1 - pos[2] + eps)
            
            if len(position_history[i]) >= 2:
                recent_pos = np.array(position_history[i][-2:])
                if np.all(np.linalg.norm(recent_pos - recent_pos[0], axis=1) < 0.001):
                    particles[i] = np.random.uniform([-1, -1, -1], [1, 1, 1])
                    particles[i] /= np.linalg.norm(particles[i])
                    trails[i] = np.full((map_f["trail_length"], 3), np.nan)
                    position_history[i] = []
                    continue

            if not (x_min <= x_proj <= x_max and y_min <= y_proj <= y_max):
                particles[i] = np.random.uniform([-1, -1, -1], [1, 1, 1])
                particles[i] /= np.linalg.norm(particles[i])
                trails[i] = np.full((map_f["trail_length"], 3), np.nan)
                position_history[i] = []
                continue

            u = interpolator_u([[x_proj, y_proj]])[0]
            v = interpolator_v([[x_proj, y_proj]])[0]
            w = interpolator_w([[x_proj, y_proj]])[0]
            vec = np.array([u, v, w])
            
            vec_norm = vec / (np.linalg.norm(vec) + eps)
            
            particles[i] += vec_norm * map_f["dt"]
            particles[i] /= np.linalg.norm(particles[i])

            position_history[i].append(particles[i].copy())
            if len(position_history[i]) > 2:
                position_history[i].pop(0)

            trails[i] = np.roll(trails[i], shift=-1, axis=0)
            trails[i][-1] = particles[i]

        scat._offsets3d = (particles[:, 0], particles[:, 1], particles[:, 2])
        segments = [trail[j:j+2] for trail in trails for j in range(len(trail)-1) 
                    if not np.any(np.isnan(trail[j:j+2]))]
        trail_lines.set_segments(segments)
        
        return scat, trail_lines

    ani = FuncAnimation(fig, update, frames=map_f["frames"], interval=map_f["interval"], 
                       blit=False, repeat=True)
    plt.gca().set_aspect('equal')
    plt.show()
