import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time

class FrameVisualizer:
    def __init__(self):
        self.vis = meshcat.Visualizer()
        
    def create_frame(self, name, scale=1.0, tube_radius=0.05):
        """Create a coordinate frame with RGB arrows for XYZ and labels."""
        # Create arrows
        for axis, color in [('x', 0xff0000), ('y', 0x00ff00), ('z', 0x0000ff)]:
            # Arrow shaft
            self.vis[f"{name}/{axis}"].set_object(
                g.Cylinder(tube_radius * scale, scale),
                g.MeshPhongMaterial(color=color))
            
            # Arrow head (cone)
            self.vis[f"{name}/{axis}/head"].set_object(
                g.Cone(tube_radius * 2 * scale, tube_radius * 4 * scale),
                g.MeshPhongMaterial(color=color))
        
        # Set orientations and positions
        self.vis[f"{name}/x"].set_transform(
            tf.rotation_matrix(np.pi/2, [0, 1, 0]) @ tf.translation_matrix([scale/2, 0, 0]))
        self.vis[f"{name}/x/head"].set_transform(
            tf.rotation_matrix(np.pi/2, [0, 1, 0]) @ tf.translation_matrix([scale, 0, 0]))
        
        self.vis[f"{name}/y"].set_transform(
            tf.rotation_matrix(-np.pi/2, [1, 0, 0]) @ tf.translation_matrix([0, scale/2, 0]))
        self.vis[f"{name}/y/head"].set_transform(
            tf.rotation_matrix(-np.pi/2, [1, 0, 0]) @ tf.translation_matrix([0, scale, 0]))
        
        self.vis[f"{name}/z"].set_transform(tf.translation_matrix([0, 0, scale/2]))
        self.vis[f"{name}/z/head"].set_transform(tf.translation_matrix([0, 0, scale]))

        # Add a small sphere at origin
        self.vis[f"{name}/origin"].set_object(
            g.Sphere(tube_radius * 1.5 * scale),
            g.MeshPhongMaterial(color=0xFFFFFF))
    
    def visualize_transform(self, name, T, parent_frame=""):
        """Visualize a homogeneous transform T relative to parent_frame."""
        path = f"{parent_frame}/{name}" if parent_frame else name
        self.create_frame(path, scale=0.5 if parent_frame else 1.0)
        self.vis[path].set_transform(T)
        
        # Visualize connecting line between frames if there's a parent
        if parent_frame:
            points = np.array([[0, 0, 0], T[:3, 3]]).T
            self.vis[f"{parent_frame}/connection_{name}"].set_object(
                g.LineSegments(
                    g.PointsGeometry(points),
                    g.MeshBasicMaterial(color=0x808080)
                ))
    
    def animate_transform(self, name, T_start, T_end, duration=2.0, fps=30, show_path=True):
        """Animate a smooth transition between two transforms."""
        R1, t1 = T_start[:3,:3], T_start[:3,3]
        R2, t2 = T_end[:3,:3], T_end[:3,3]
        n_frames = int(duration * fps)
        
        # Create path visualization if requested
        if show_path:
            path_points = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            t = i / (n_frames - 1)
            # SLERP for rotation
            trace_val = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)
            theta = np.arccos(trace_val)
            
            if np.abs(theta) < 1e-10:
                R = R1
            else:
                R = R1 @ (np.eye(3) + np.sin(theta * t)/np.sin(theta) * 
                         (R1.T @ R2 - np.eye(3)))
            
            # Linear interpolation for translation
            trans = t1 + t * (t2 - t1)
            
            if show_path:
                path_points[:, i] = trans
            
            # Construct interpolated transform
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = trans
            
            self.vis[name].set_transform(T)
            
            # Update path visualization
            if show_path and i > 0:
                self.vis[f"{name}_path"].set_object(
                    g.LineSegments(
                        g.PointsGeometry(path_points[:, :i+1]),
                        g.MeshBasicMaterial(color=0xFFFF00)
                    ))
            
            time.sleep(1/fps)

# Example usage
if __name__ == "__main__":
    viz = FrameVisualizer()
    
    # Create world frame
    viz.create_frame("world")
    
    # Create and animate multiple transforms
    # First transform: 45° rotation about Z and translation
    T1 = tf.translation_matrix([1, 1, 0.5]) @ tf.rotation_matrix(np.pi/4, [0, 0, 1])
    viz.visualize_transform("frame1", T1, "world")
    
    # Second transform: 90° rotation about X and different translation
    T2 = tf.translation_matrix([1, -1, 1]) @ tf.rotation_matrix(np.pi/2, [1, 0, 0])
    time.sleep(1)  # Pause to see initial position
    
    # Animate between transforms
    viz.animate_transform("frame1", T1, T2, duration=3.0)
    
    # Keep visualization window open
    input("Press Enter to close...")