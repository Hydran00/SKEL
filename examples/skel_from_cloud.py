import argparse
import os
import pickle

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
import trimesh
import torch
import sys
from tqdm import tqdm
from skel.alignment.losses import compute_scapula_loss

from skel.skel_model import SKEL

sys.path.append("../")


#!/usr/bin/env python3
import torch
import numpy as np
import open3d as o3d
from chamferdist import ChamferDistance
import os
import copy

NUM_BETAS = 10
DEVICE = "cuda:0"

print("CUDA available:", torch.cuda.is_available())


class SKELModelOptimizer:
    def __init__(self, skel_model, device=DEVICE):
        self.skel_model = skel_model
        self.betas = torch.zeros(
            (1, NUM_BETAS), dtype=torch.float32, requires_grad=True, device=DEVICE
        )
        self.poses = torch.zeros(
            (1, self.skel_model.num_q_params),
            dtype=torch.float32,
            requires_grad=True,
            device=DEVICE,
        )
        self.global_orient = torch.zeros(
            (1, 3), dtype=torch.float32, requires_grad=True, device=DEVICE
        )
        self.global_position = torch.zeros(
            (1, 3), dtype=torch.float32, requires_grad=True, device=DEVICE
        )
        self.chamfer_distance = ChamferDistance().to(DEVICE)
        self.skel_skin_vis_offset = [2.0, 0.0, 0.0]
        # Open3D visualization setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.mesh = o3d.geometry.TriangleMesh()
        self.point_cloud = None


    def optimize_model(self, target_pc):
        self.target_pc_tensor = self.point_cloud_to_tensor(target_pc).to(DEVICE)
        self.point_cloud = target_pc
        self.init_visualization(target_pc)

        # Optimization steps
        mask = torch.zeros_like(self.poses)
        mask[:, :3] = 1.0  # Only allow gradients on rotation
        self.optimize(
            [self.global_position],
            lr=0.05,
            loss_type="transl",
            num_iterations=200,
            tolerance_change=1e-5,
        )
        self.optimize(
            [self.poses],
            lr=0.01,
            loss_type="rot",
            num_iterations=200,
            tolerance_change=1e-4,
            mask=mask,  # Pass the mask to be applied inside optimize()
        )

        # self.optimize(
        #     [self.global_position],
        #     lr=0.05,
        #     loss_type="transl",
        #     num_iterations=100,
        #     tolerance_change=1e-6,
        # )
        self.optimize(
            [self.poses],
            lr=0.02,
            loss_type="transl",
            num_iterations=200,
            tolerance_change=1e-4,
        )

        
        self.optimize([self.betas], lr=0.002, loss_type="shape", num_iterations=400, tolerance_change=1e-6)

        return (
            self.betas.detach(),
            self.poses.detach(),
            self.global_position.detach(),
        )

    def point_cloud_to_tensor(self, point_cloud):
        return (
            torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)
            if isinstance(point_cloud, o3d.geometry.PointCloud)
            else torch.tensor(point_cloud, dtype=torch.float32)
        )

    def compute_loss(self, loss_type, generated_pc):
        # gen_pc_tensor = self.point_cloud_to_tensor(generated_pc).to(DEVICE)
        gen_pc_tensor = generated_pc
        loss = self.chamfer_distance(
            gen_pc_tensor, self.target_pc_tensor.unsqueeze(0), reverse=True
        )
        # print(f"{loss_type} loss: {loss.item():.6f}")
        return loss

    def get_skel(self):
        return self.skel_model.forward(
            poses=self.poses,
            betas=self.betas,
            trans=self.global_position,
            poses_type="skel",
            skelmesh=True,
        )

    def optimize(self, params, lr, loss_type, num_iterations, tolerance_change, mask=None):
        optimizer = torch.optim.AdamW(params, lr=lr)
        last_loss = 1e6
        pbar = tqdm(range(num_iterations))
        for i in pbar:
            optimizer.zero_grad()
            skel_output = self.get_skel()
            skin_vertices = skel_output.skin_verts
            skel_vertices = skel_output.skel_verts
            loss1 = self.compute_loss(loss_type, skin_vertices)
            loss2 = 1.0 * compute_scapula_loss(self.poses)
            skin_vertices_vis = skin_vertices.cpu().detach().clone().numpy()
            skel_vertices_vis = skel_vertices.cpu().detach().clone().numpy()
            loss = loss1 + loss2
            loss.backward()
            # Apply the mask to prevent updates on non-rotation parameters
            if mask is not None and self.poses.grad is not None:
                self.poses.grad *= mask 
            loss_change = abs(last_loss - loss.item())
            if  loss_change < tolerance_change:
                print(f"Converged at iteration {i}, Loss: {loss.item():.6f}")
                break 
            last_loss = loss.item()
            optimizer.step()
            if i % 1 == 0:  # Update visualization every 10 iterations
                # print(f"Iteration {i}, Loss: {loss.item():.6f}")
                pbar.set_description(f"Iteration {i}, Loss: {loss.item():.2f}, Loss change: {loss_change:.6f}")
                self.update_visualization(skin_vertices_vis, skel_vertices_vis)

        print(f"Final {loss_type} loss: {loss.item():.6f}")

    def init_visualization(self, target_pc):
        """Initializes Open3D visualization with the input point cloud and SMPL model."""
        self.vis.clear_geometries()
        self.reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.reference_frame)
        # Add target point cloud
        target_pc_skin_vis = copy.deepcopy(target_pc)
        target_pc_skel_vis = copy.deepcopy(target_pc)
        target_pc_skel_vis_vertices = np.asarray(target_pc_skel_vis.points) + self.skel_skin_vis_offset
        target_pc_skel_vis.points = o3d.utility.Vector3dVector(target_pc_skel_vis_vertices)
        target_pc_skin_vis.paint_uniform_color([0, 0, 1])  # Blue color
        self.vis.add_geometry(target_pc_skin_vis)
        self.vis.add_geometry(target_pc_skel_vis)

        # Initialize the skin mesh
        skel_output = self.get_skel()
        skin_vertices = skel_output.skin_verts.cpu().detach().numpy()
        skin_faces = self.skel_model.skin_f.cpu().detach().numpy()
        # Initialize the skeleton mesh
        skel_vertices = skel_output.skel_verts.detach().clone().cpu().numpy() + self.skel_skin_vis_offset
        skel_faces = self.skel_model.skel_f.cpu().numpy().copy()

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(skin_vertices[0])
        self.mesh.triangles = o3d.utility.Vector3iVector(skin_faces)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([1, 0.5, 0])  # Orange color for SMPL model
        self.vis.add_geometry(self.mesh)

        self.skel_mesh = o3d.geometry.TriangleMesh()
        self.skel_mesh.vertices = o3d.utility.Vector3dVector(skel_vertices[0])
        self.skel_mesh.triangles = o3d.utility.Vector3iVector(skel_faces)
        self.skel_mesh.compute_vertex_normals()
        self.vis.add_geometry(self.skel_mesh)

        self.vis.poll_events()
        self.vis.update_renderer()

    def update_visualization(self, skin_vertices, skel_vertices):
        """Updates the Open3D visualizer with new SMPL mesh."""
        self.mesh.vertices = o3d.utility.Vector3dVector(skin_vertices[0])
        self.mesh.compute_vertex_normals()
        self.skel_mesh.vertices = o3d.utility.Vector3dVector(skel_vertices[0]+ self.skel_skin_vis_offset) 
        self.skel_mesh.compute_vertex_normals()

        self.vis.update_geometry(self.mesh)
        self.vis.update_geometry(self.skel_mesh)
        self.vis.poll_events()
        self.vis.update_renderer()


    def close_visualization(self):
        """Closes Open3D visualizer."""
        self.vis.destroy_window()


def opt():
    print("Loading point cloud...")
    pc = o3d.io.read_point_cloud("merged-filtered.ply")

    ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0]
    )
    def radians_from_deg(rad):
        return rad * np.pi / 180
    rot = pc.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    pc.rotate(rot, center=(0, 0, 0))
    rot = pc.get_rotation_matrix_from_xyz((0, np.pi/6, 0))
    pc.rotate(rot, center=(0, 0, 0))
    rot = pc.get_rotation_matrix_from_xyz((-np.pi/10, 0, 0))
    pc.rotate(rot, center=(0, 0, 0))
    rot = pc.get_rotation_matrix_from_xyz((0, radians_from_deg(20), 0))
    pc.rotate(rot, center=(0, 0, 0))
    rot = pc.get_rotation_matrix_from_xyz((radians_from_deg(10),0,0))
    pc.rotate(rot, center=(0, 0, 0))
    pc.translate([1.0, 0, 1.3])

    # keep points with z > -0.4
    points = np.asarray(pc.points)
    idx = np.where(points[:, 2] > -0.4)
    pc.points = o3d.utility.Vector3dVector(points[idx])
    pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx]) 


    o3d.visualization.draw_geometries([pc, ref_frame])

    

    # downsample
    pc = pc.voxel_down_sample(voxel_size=0.02)

    skel_model = SKEL("male").to(DEVICE)
    optimizer = SKELModelOptimizer(skel_model)

    print("Optimizing model...")
    pc_tmp = copy.deepcopy(pc)
    betas, body_pose, global_pos = optimizer.optimize_model(pc_tmp)

    # visualize the optimized model
    skel_output = optimizer.get_skel()
    vertices = skel_output.skin_verts.cpu().detach().numpy()
    faces = skel_model.skin_f.cpu().detach().numpy()
    optimized_mesh = o3d.geometry.TriangleMesh()
    optimized_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    optimized_mesh.triangles = o3d.utility.Vector3iVector(faces)
    optimized_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([pc, optimized_mesh, ref_frame])

    print("Optimization complete!")
    print("Optimized betas:", betas)
    print("Optimized body pose:", body_pose)
    print("Optimized global position:", global_pos)

    optimizer.close_visualization()


if __name__ == "__main__":
    opt()
