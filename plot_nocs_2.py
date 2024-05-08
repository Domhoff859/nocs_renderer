import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from matplotlib.animation import FuncAnimation

def plot_mesh(ax, mesh_path, angle):
    # Load the mesh
    mesh = trimesh.load(mesh_path)

    max_extent = np.max(mesh.vertices, axis=0)
    min_extent = np.min(mesh.vertices, axis=0)
    scale_factor = 1 / np.max(max_extent - min_extent)
    translate_vector = -min_extent * scale_factor

    # Normalize the mesh to fit within a 1x1x1 cube
    mesh.apply_transform(trimesh.transformations.scale_and_translate(scale=scale_factor, translate=translate_vector))

    target_center = np.array([0.5, 0.5, 0.38])
    current_center = np.mean(mesh.vertices, axis=0)
    center_translation = target_center - current_center

    # Apply the translation to move the center of the mesh
    mesh.apply_translation(center_translation)

    # Extract face colors from the PLY file
    face_colors = mesh.visual.face_colors / 255.0

    # Plot the mesh faces with individual colors
    mpl_mesh = Poly3DCollection(mesh.vertices[mesh.faces], alpha=1.0)
    mpl_mesh.set_facecolor(face_colors)

    ax.add_collection3d(mpl_mesh)

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Remove grid
    ax.grid(False)

    # Remove axis labels
    ax.set_axis_off()

    ax.view_init(elev=0., azim=angle)  # Set the viewpoint angle with elevation

def save_image(ax, mesh_path, angle, frame):
    ax.cla()
    plot_mesh(ax, mesh_path, angle)
    plt.savefig(f"frame_{frame:03d}.png")

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mesh_path = "meshes/obj_000005.ply"
frames = 24  # Number of frames for the animation, 15 degrees rotation per frame

# Save images after every 15 degrees rotation
for frame in range(frames):
    angle = 15 * frame  # 15 degrees rotation per frame
    save_image(ax, mesh_path, angle, frame + 1)

plt.close()