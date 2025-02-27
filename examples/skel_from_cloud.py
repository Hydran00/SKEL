# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import pickle

os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
import trimesh
import torch
import sys

from skel.skel_model import SKEL

sys.path.append("../")

from skel.alignment.aligner import SkelFitter

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
        # self.global_orient = torch.zeros(
        #     (1, 3), dtype=torch.float32, requires_grad=True, device=DEVICE
        # )
        self.global_position = torch.zeros(
            (1, 3), dtype=torch.float32, requires_grad=True, device=DEVICE
        )
        self.chamfer_distance = ChamferDistance().to(DEVICE)

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
        self.optimize(
            [self.global_position],
            lr=0.1,
            loss_type="transl",
            num_iterations=200,
        )
        # self.optimize(
        #     [self.global_orient],
        #     lr=0.1,
        #     loss_type="transl",
        #     num_iterations=200,
        # )
        self.optimize(
            [self.poses],
            lr=0.01,
            loss_type="transl",
            num_iterations=200,
        )
        self.optimize([self.betas], lr=0.002, loss_type="shape", num_iterations=400)

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
        gen_pc_tensor = self.point_cloud_to_tensor(generated_pc).to(DEVICE)
        loss = self.chamfer_distance(
            gen_pc_tensor, self.target_pc_tensor.unsqueeze(0), reverse=True
        )
        print(f"{loss_type} loss: {loss.item():.6f}")
        return loss

    def get_skel(self):
        return self.skel_model.forward(
            poses=self.poses,
            betas=self.betas,
            trans=self.global_position,
            poses_type="skel",
            skelmesh=True,
        )

    def optimize(self, params, lr, loss_type, num_iterations):
        optimizer = torch.optim.LBFGS(params, lr=lr)
        for i in range(num_iterations):
            optimizer.zero_grad()
            skel_output = self.get_skel()
            vertices = skel_output.skin_verts
            loss = self.compute_loss(loss_type, vertices)
            # vertices_vis = copy.deepcopy(vertices.cpu().detach().numpy())
            # print("global_position grad:", self.global_position.grad)
            loss.backward()
            optimizer.step()

            if i % 3 == 0:  # Update visualization every 10 iterations
                print(f"Iteration {i}, Loss: {loss.item():.6f}")
                # self.update_visualization(vertices_vis)

        print(f"Final {loss_type} loss: {loss.item():.6f}")

    def init_visualization(self, target_pc):
        """Initializes Open3D visualization with the input point cloud and SMPL model."""
        self.vis.clear_geometries()
        self.reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.reference_frame)
        # Add target point cloud
        target_pc.paint_uniform_color([0, 0, 1])  # Blue color
        self.vis.add_geometry(target_pc)

        # Initialize the SMPL mesh
        skel_output = self.get_skel()
        # skel_vertices = skel_output.skel_verts.cpu().numpy()
        skin_vertices = skel_output.skin_verts.cpu().detach().numpy()
        # skel_faces = self.skel_model.skel_f.cpu().numpy().copy()
        skin_faces = self.skel_model.skin_f.cpu().detach().numpy()

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(skin_vertices[0])
        self.mesh.triangles = o3d.utility.Vector3iVector(skin_faces)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([1, 0.5, 0])  # Orange color for SMPL model
        self.vis.add_geometry(self.mesh)

        self.vis.poll_events()
        self.vis.update_renderer()

    def update_visualization(self, vertices):
        """Updates the Open3D visualizer with new SMPL mesh."""
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
        self.mesh.compute_vertex_normals()

        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close_visualization(self):
        """Closes Open3D visualizer."""
        self.vis.destroy_window()


def opt():
    print("Loading point cloud...")
    pc = o3d.io.read_point_cloud("merged-filtered.ply")

    ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([pc, ref_frame])

    # downsample
    pc = pc.voxel_down_sample(voxel_size=0.005)

    skel_model = SKEL("male").to(DEVICE)
    optimizer = SKELModelOptimizer(skel_model)

    print("Optimizing model...")
    pc_tmp = copy.deepcopy(pc)
    betas, body_pose, global_pos = optimizer.optimize_model(pc_tmp)

    # visualize the optimized model
    skel_output = optimizer.get_skel()
    vertices = skel_output.vertices[0].cpu().detach().numpy()
    optimized_mesh = o3d.geometry.TriangleMesh()
    optimized_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    optimized_mesh.triangles = o3d.utility.Vector3iVector(skel_model.faces)
    optimized_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([pc, optimized_mesh, ref_frame])

    print("Optimization complete!")
    print("Optimized betas:", betas)
    print("Optimized body pose:", body_pose)
    print("Optimized global position:", global_pos)

    optimizer.close_visualization()


if __name__ == "__main__":
    opt()
