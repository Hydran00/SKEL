import argparse
import os
import pickle
import zmq
import numpy as np
import open3d as o3d
import torch
import copy
import sys
from tqdm import tqdm
from skel.alignment.losses import compute_scapula_loss
from skel.skel_model import SKEL
from chamferdist import ChamferDistance
import os
import json

# Environment setup
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

# ZMQ Communication Settings
ZMQ_PORT_RECV = "5555"  # Port to receive point cloud from node1
ZMQ_PORT_SEND = "5556"  # Port to send optimized results back to node1

# Constants
NUM_BETAS = 10
DEVICE = "cuda:0"
print("CUDA available:", torch.cuda.is_available())

# Initialize ZMQ Context
context = zmq.Context()

# ZMQ Subscriber (Receives point cloud)
socket_recv = context.socket(zmq.SUB)
socket_recv.connect(f"tcp://localhost:{ZMQ_PORT_RECV}")
socket_recv.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages

# ZMQ Publisher (Sends back optimized results)
socket_send = context.socket(zmq.PUB)
socket_send.bind(f"tcp://*:{ZMQ_PORT_SEND}")


class SKELModelOptimizer:
    def __init__(self, skel_model, global_position_init, global_rotation_init, device=DEVICE):
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
        with torch.no_grad():
            self.poses[:, :3] = global_rotation_init
        self.global_position = torch.tensor(
            global_position_init, dtype=torch.float32, requires_grad=True, device=DEVICE
        )
        self.chamfer_distance = ChamferDistance().to(DEVICE)
        self.skel_skin_vis_offset = [2.0, 0.0, 0.0]
        self.mesh = o3d.geometry.TriangleMesh()
        self.point_cloud = None
        self.global_orient_mask = torch.zeros(
            (1, self.skel_model.num_q_params), dtype=torch.float32, device=DEVICE
        )
        self.global_orient_mask[:, :3] = 1.0

    def optimize_model(self, target_pc):
        self.target_pc_tensor = self.point_cloud_to_tensor(target_pc).to(DEVICE)
        self.point_cloud = target_pc
        self.init_visualization(target_pc)

        # Optimization steps
        mask = torch.zeros_like(self.poses)
        mask[:, :3] = 1.0  # Only allow gradients on rotation
        self.optimize(
            [self.global_position],
            lr=0.1,
            loss_type="transl",
            num_iterations=100,
            # tolerance_change=1e-5,
            tolerance_change=1e-4,
        )
        print("Optimized global position")
        self.optimize(
            [self.poses],
            lr=0.03,
            loss_type="rot",
            num_iterations=100,
            tolerance_change=1e-4,
            mask=mask,  # Pass the mask to be applied inside optimize()
        )
        print("Optimized global rotation")
        self.optimize(
            [self.global_position,self.poses],
            lr=0.01,
            loss_type="rot",
            num_iterations=100,
            tolerance_change=1e-4,
            mask=mask,  # Pass the mask to be applied inside optimize()
        )
        # print optimized position and rotation
        print("Optimized global position:", self.global_position.cpu().detach().numpy())
        print("Optimized rotation:", self.poses[:, :3].cpu().detach().numpy())
        self.optimize(
            [self.poses],
            lr=0.02,
            loss_type="transl",
            num_iterations=200,
            tolerance_change=1e-4,
        )

        self.optimize(
            [self.betas],
            lr=0.2,
            loss_type="shape",
            num_iterations=400,
            tolerance_change=1e-5,
        )

        # return
        #     self.betas.detach(),
        #     self.poses.detach(),
        #     self.global_position.detach(),
        # )
        return self.get_skel()

    def point_cloud_to_tensor(self, point_cloud):
        return (
            torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)
            if isinstance(point_cloud, o3d.geometry.PointCloud)
            else torch.tensor(point_cloud, dtype=torch.float32)
        )

    def compute_loss(self, loss_type, generated_pc):
        if loss_type == "transl" or loss_type == "rot":
            return self.chamfer_distance(
                generated_pc, self.target_pc_tensor.unsqueeze(0), reverse=True
            )
        elif loss_type == "shape":
            return 0.000 * self.betas.pow(2).sum() + self.chamfer_distance(
                generated_pc, self.target_pc_tensor.unsqueeze(0), reverse=True
            )
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

    def get_skel(self):
        return self.skel_model.forward(
            poses=self.poses,
            betas=self.betas,
            trans=self.global_position,
            poses_type="skel",
            skelmesh=True,
        )

    def optimize(
        self, params, lr, loss_type, num_iterations, tolerance_change, mask=None
    ):
        optimizer = torch.optim.AdamW(params, lr=lr)
        last_loss = 1e6
        pbar = tqdm(range(num_iterations))
        for i in pbar:
            optimizer.zero_grad()
            skel_output = self.get_skel()
            skin_vertices = skel_output.skin_verts
            # skel_vertices = skel_output.skel_verts
            loss1 = self.compute_loss(loss_type, skin_vertices)
            loss2 = 1.0 * compute_scapula_loss(self.poses)
            skin_vertices_vis = skin_vertices.cpu().detach().clone().numpy()
            # skel_vertices_vis = skel_vertices.cpu().detach().clone().numpy()
            loss = loss1 + loss2
            loss.backward()
            # Apply the mask to prevent updates on non-rotation parameters
            if mask is not None and self.poses.grad is not None:
                self.poses.grad *= mask
            loss_change = abs(last_loss - loss.item())
            if loss_change < tolerance_change:
                print(f"Converged at iteration {i}, Loss: {loss.item():.4f}, Loss change: {loss_change:.6f}")
                break
            last_loss = loss.item()
            optimizer.step()
            if i % 1 == 0:  # Update visualization every 10 iterations
                # print(f"Iteration {i}, Loss: {loss.item():.6f}")
                pbar.set_description(
                    f"Iteration {i}, Chamfer dist loss: {loss1.item():.2f}, Scapula loss: {loss2.item():.2f}, Total loss change: {loss_change:.6f}"
                )
                self.update_visualization(skin_vertices_vis)

        print(f"Final {loss_type} loss: {last_loss:.6f}")

    def init_visualization(self, target_pc):
        # Open3D visualization setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.clear_geometries()
        ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]
        )
        self.vis.add_geometry(ref_frame)
        target_pc.paint_uniform_color([0, 0, 1])
        self.vis.add_geometry(target_pc)
        skel_output = self.get_skel()
        skin_vertices = skel_output.skin_verts.cpu().detach().numpy()
        skin_faces = self.skel_model.skin_f.cpu().detach().numpy()
        self.mesh.vertices = o3d.utility.Vector3dVector(skin_vertices[0])
        self.mesh.triangles = o3d.utility.Vector3iVector(skin_faces)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([1, 0.5, 0])
        self.vis.add_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_visualization(self, skin_vertices):
        self.mesh.vertices = o3d.utility.Vector3dVector(skin_vertices[0])
        self.mesh.compute_vertex_normals()
        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close_visualization(self):
        self.vis.destroy_window()


def send_optimized_results(data, output_folder):
    
    # use json
    content = pickle.dumps(data)
    socket_send.send(content)
    if output_folder is not None:
        with open(output_folder + "optimized_skel_params.json", "w") as f:
            json.dump(data, f, indent=4)
    print("Optimized results sent back.")


def receive_point_cloud():
    print("Waiting for point cloud from node1...")
    msg = socket_recv.recv()
    point_cloud_data = pickle.loads(msg)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data["points"])
    point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_data["colors"])
    return point_cloud


def main(args=None):

    # get arguments
    parser = argparse.ArgumentParser(description="Skeleton fitting from point cloud")
    parser.add_argument(
        "--from_path",
        type=bool,
        default=False,
        help="Load point cloud from file",
    )
    parser.add_argument(
        "--use_initialization",
        type=bool,
        default=False,
        help="Use the last found parameters as initialization",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.path.expanduser("~") + "/temp/skel_fitting/",
        help="Output folder to save optimized parameters",
    )
    args = parser.parse_args(args)

    output_folder = args.output_folder
    print("\n\n \033[92m Output folder: ", output_folder, "\033[0m \n\n")


    # Initialize SKEL model and optimizer
    skel_model = SKEL("male").to(DEVICE)
    global_position = torch.tensor([[0.6195589, 0.17820874, 0.10494429]], device=DEVICE)
    global_rotation = torch.tensor([3.11, 0.0, 3.0002122], device=DEVICE)
    optimizer = SKELModelOptimizer(skel_model, global_position, global_rotation)
 
    if args.use_initialization:
        with open(output_folder + "optimized_skel_params.pkl", "rb") as f:
            # params = pickle.load(f)
            # optimizer.betas = torch.tensor(params["betas"][:,:,0], device=DEVICE)
            # optimizer.poses = torch.tensor(params["body_pose"][:,:,0], device=DEVICE)
            # optimizer.global_position = torch.tensor(params["global_pos"][:,0], device=DEVICE)
            print("Using last found parameters as initialization.")    

    while True:
        print("Waiting for point cloud...")
        if args.from_path:
            print("Loading point cloud from file...")
            pc = o3d.io.read_point_cloud(os.environ["OUTPUT_FOLDER"]+"/icp_output/merged.ply")
        else:
            pc = receive_point_cloud()

        ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0]
        )

        # downsample
        pc = pc.voxel_down_sample(voxel_size=0.01)

        # optimize skel model
        optimized_skel = optimizer.optimize_model(pc)
        # optimized_skel = optimizer.get_skel()

        # send optimized results back   
        skin_vertices = optimized_skel.skin_verts.cpu().detach().numpy()
        skin_faces = optimizer.skel_model.skin_f.cpu().detach().numpy()
        skin_mesh = o3d.geometry.TriangleMesh()
        skin_mesh.vertices = o3d.utility.Vector3dVector(skin_vertices[0])
        skin_mesh.triangles = o3d.utility.Vector3iVector(skin_faces)
        skel_verts = optimized_skel.skel_verts.cpu().detach().numpy()
        skel_faces = optimizer.skel_model.skel_f.cpu().detach().numpy()
        skel_mesh = o3d.geometry.TriangleMesh()
        skel_mesh.vertices = o3d.utility.Vector3dVector(skel_verts[0])
        skel_mesh.triangles = o3d.utility.Vector3iVector(skel_faces)
        results = {
            "skel_verts": skel_verts.tolist(),
            "skel_faces": skel_faces.tolist(),
            "skin_verts": skin_vertices.tolist(),
            "skin_faces": skin_faces.tolist(),
            "betas": optimized_skel.betas.cpu().detach().numpy().tolist(),
            "body_pose": optimized_skel.poses.cpu().detach().numpy().tolist(),
            "global_pos": optimized_skel.trans.cpu().detach().numpy().tolist(),
            "joints_ori": optimized_skel.joints_ori.cpu().detach().numpy().tolist(),
            "joints": optimized_skel.joints.cpu().detach().numpy().tolist(),
        }
        print(results["betas"])
        send_optimized_results(results, output_folder)
        o3d.io.write_triangle_mesh(output_folder + "optimized_skel.ply", skel_mesh)
        o3d.io.write_triangle_mesh(output_folder + "optimized_skin.ply", skin_mesh)
        print("Optimization cycle complete.")
        optimizer.close_visualization()
        exit(0)


if __name__ == "__main__":
    main()
