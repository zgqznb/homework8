

import os
import sys
import types
import argparse
import traceback

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# smplx 和 smplx.lbs 需要你的环境里已经有。
# 你的 taichi 实验环境如果已经集成这些库，就不用额外安装。
import smplx
from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 0. 兼容老 SMPL pkl：不用真实安装 chumpy
# ============================================================

class _ChumpyArrayShim:
    """Minimal pickle shim for old SMPL files that stored arrays as chumpy.Ch."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    def _array(self):
        # 老 SMPL 文件里常见字段：r 或 x
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x

        # 兜底：递归找第一个 numpy 数组
        arr = _find_first_numpy_array(self)
        if arr is not None:
            return arr

        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]


def _find_first_numpy_array(obj, max_depth=8):
    seen = set()

    def rec(x, depth):
        if depth > max_depth:
            return None
        obj_id = id(x)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        if isinstance(x, np.ndarray):
            return x

        if hasattr(x, "toarray"):
            try:
                return np.asarray(x.toarray())
            except Exception:
                pass

        if isinstance(x, dict):
            for v in x.values():
                out = rec(v, depth + 1)
                if out is not None:
                    return out

        if isinstance(x, (list, tuple)):
            for v in x:
                out = rec(v, depth + 1)
                if out is not None:
                    return out

        if hasattr(x, "__dict__"):
            for v in x.__dict__.values():
                out = rec(v, depth + 1)
                if out is not None:
                    return out

        return None

    return rec(obj, 0)


def install_chumpy_pickle_shim():
    """Allow pickle.load to read legacy SMPL .pkl files without installing chumpy."""
    if "chumpy.ch" in sys.modules:
        return

    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")

    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim

    # 少量兜底属性，避免某些 pickle/旧代码访问 chumpy.Ch
    chumpy_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module

    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


# ============================================================
# 1. 通用工具
# ============================================================

def make_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_script_path(path: str):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def normalize_model_path_for_smplx(model_path: str):
    """
    支持以下三种输入：
    1. models
    2. models/smpl
    3. models/smpl/SMPL_NEUTRAL.pkl

    smplx.create 最稳的是传根目录 models。
    """
    model_path = resolve_script_path(model_path)
    model_path = os.path.abspath(model_path)

    if os.path.isfile(model_path):
        parent = os.path.dirname(model_path)
        if os.path.basename(parent).lower() == "smpl":
            return os.path.dirname(parent)
        return parent

    if os.path.isdir(model_path):
        # 如果直接传了 models/smpl，并且里面有 SMPL_NEUTRAL.pkl，则返回上一级 models
        if os.path.basename(model_path).lower() == "smpl":
            if os.path.exists(os.path.join(model_path, "SMPL_NEUTRAL.pkl")):
                return os.path.dirname(model_path)
        return model_path

    raise FileNotFoundError(f"模型路径不存在：{model_path}")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ============================================================
# 2. 3D 可视化：透视 + 3/4 视角 + 阴影
# ============================================================

def smpl_to_plot_coords(points: np.ndarray):
    """
    SMPL 通常 y 轴向上，matplotlib 3D 默认 z 轴向上。
    这里把原始 (x, y, z) 映射为绘图坐标 (x, z, y)。
    """
    points = np.asarray(points)
    return points[:, [0, 2, 1]]


def set_axes_equal(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def get_face_colors_from_vertex_scalar(vertex_scalar: np.ndarray, faces: np.ndarray, cmap_name="viridis"):
    scalar = np.asarray(vertex_scalar, dtype=np.float64).reshape(-1)
    scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    face_scalar = scalar[faces].mean(axis=1)
    cmap = plt.get_cmap(cmap_name)
    return cmap(face_scalar)


def get_face_colors_from_joint_weights(lbs_weights: np.ndarray, faces: np.ndarray):
    """
    全关节主导权重图：
    - 色相表示主要由哪个关节控制；
    - 明暗表示该主导权重强弱。
    """
    face_weights = lbs_weights[faces].mean(axis=1)
    dominant_joint = np.argmax(face_weights, axis=1)
    dominant_weight = np.max(face_weights, axis=1)

    num_joints = lbs_weights.shape[1]
    palette = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, num_joints, endpoint=False))
    face_colors = palette[dominant_joint]

    strength = 0.35 + 0.65 * dominant_weight
    face_colors[:, :3] *= strength[:, None]
    face_colors[:, :3] += (1.0 - strength[:, None]) * 0.88
    face_colors[:, 3] = 1.0
    return face_colors


def shade_face_colors(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray):
    """给三角面加简单 Lambert 光照，让结果看起来明显是 3D。"""
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)
    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded


def draw_skeleton(ax, plot_joints, parents=None):
    ax.scatter(
        plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2],
        c="white", s=13, depthshade=False,
        edgecolors="black", linewidths=0.35,
        zorder=10,
    )

    if parents is not None:
        parents = np.asarray(parents, dtype=np.int64)
        for i in range(1, min(len(parents), len(plot_joints))):
            p = int(parents[i])
            if p < 0 or p >= len(plot_joints):
                continue
            ax.plot(
                [plot_joints[p, 0], plot_joints[i, 0]],
                [plot_joints[p, 1], plot_joints[i, 1]],
                [plot_joints[p, 2], plot_joints[i, 2]],
                color="black", linewidth=0.8, alpha=0.65,
            )


def draw_mesh(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray = None,
    parents: np.ndarray = None,
    vertex_scalar: np.ndarray = None,
    face_colors: np.ndarray = None,
    title: str = "",
    elev: float = 12,
    azim: float = 108,
    zoom: float = 1.0,
):
    plot_vertices = smpl_to_plot_coords(vertices)
    plot_joints = None if joints is None else smpl_to_plot_coords(joints)

    if face_colors is not None:
        face_colors = face_colors.copy()
    elif vertex_scalar is None:
        # 肉色基础色
        face_colors = np.tile(np.array([[0.82, 0.67, 0.52, 1.0]]), (faces.shape[0], 1))
    else:
        face_colors = get_face_colors_from_vertex_scalar(vertex_scalar, faces)

    face_colors = shade_face_colors(plot_vertices, faces, face_colors)

    mesh = Poly3DCollection(
        plot_vertices[faces],
        facecolors=face_colors,
        linewidths=0.035,
        edgecolors=(0.0, 0.0, 0.0, 0.055),
    )
    ax.add_collection3d(mesh)

    if plot_joints is not None:
        draw_skeleton(ax, plot_joints, parents=parents)

    set_axes_equal(ax, plot_vertices)

    # 稍微拉近镜头
    try:
        ax.dist = 7.5 / zoom  # 旧版 matplotlib 有效，新版可能忽略
    except Exception:
        pass

    # 透视投影。老师参考图的 3D 感主要来自 3/4 视角 + 透视 + 阴影。
    try:
        ax.set_proj_type("persp", focal_length=0.85)
    except TypeError:
        ax.set_proj_type("persp")
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=2)


def save_single_figure(path, vertices, faces, joints=None, parents=None, vertex_scalar=None, title=""):
    fig = plt.figure(figsize=(5.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(
        ax,
        vertices,
        faces,
        joints=joints,
        parents=parents,
        vertex_scalar=vertex_scalar,
        title=title,
        elev=12,
        azim=108,
        zoom=1.08,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_comparison_grid(path, data_dict, faces, parents):
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(221, projection="3d")
    draw_mesh(
        ax1,
        data_dict["v_template"],
        faces,
        joints=data_dict["J_template"],
        parents=parents,
        vertex_scalar=data_dict["weight_scalar"],
        title="(a) Template + LBS Weights",
    )

    ax2 = fig.add_subplot(222, projection="3d")
    draw_mesh(
        ax2,
        data_dict["v_shaped"],
        faces,
        joints=data_dict["J_shaped"],
        parents=parents,
        title="(b) Shape Blend + Joint Regression",
    )

    ax3 = fig.add_subplot(223, projection="3d")
    draw_mesh(
        ax3,
        data_dict["v_posed"],
        faces,
        joints=data_dict["J_shaped"],
        parents=parents,
        vertex_scalar=data_dict["pose_offset_norm"],
        title="(c) Pose Blend Shapes",
    )

    ax4 = fig.add_subplot(224, projection="3d")
    draw_mesh(
        ax4,
        data_dict["verts"],
        faces,
        joints=data_dict["J_transformed"],
        parents=parents,
        title="(d) Final LBS Result",
    )

    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_all_joint_weights_figure(path, vertices, faces, joints, parents, lbs_weights):
    fig = plt.figure(figsize=(7.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(
        ax,
        vertices,
        faces,
        joints=joints,
        parents=parents,
        face_colors=get_face_colors_from_joint_weights(lbs_weights, faces),
        title="All Joint LBS Weights",
        elev=12,
        azim=108,
        zoom=1.08,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 3. 构造示例 shape / pose
# ============================================================

def build_demo_shape(device, dtype, num_betas=10):
    betas = torch.zeros((1, num_betas), dtype=dtype, device=device)
    # 设置几个非零 beta，让体型变化明显一些
    if num_betas >= 1:
        betas[0, 0] = 2.0
    if num_betas >= 2:
        betas[0, 1] = -1.2
    if num_betas >= 3:
        betas[0, 2] = 0.8
    if num_betas >= 4:
        betas[0, 3] = 0.35
    return betas


def build_demo_pose(device, dtype):
    """构造一个非零姿态，让 LBS 最终结果明显不是 T-pose。"""
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

    joint_names = {
        "left_hip": 1,
        "right_hip": 2,
        "spine1": 3,
        "left_knee": 4,
        "right_knee": 5,
        "spine2": 6,
        "spine3": 9,
        "left_shoulder": 16,
        "right_shoulder": 17,
        "left_elbow": 18,
        "right_elbow": 19,
    }

    def set_joint_pose(name, axis_angle):
        start = (joint_names[name] - 1) * 3
        body_pose[0, start:start + 3] = torch.tensor(axis_angle, dtype=dtype, device=device)

    # 上肢：略抬手，肘部微弯
    set_joint_pose("left_shoulder", [0.0, 0.0, 0.45])
    set_joint_pose("right_shoulder", [0.0, 0.0, -0.45])
    set_joint_pose("left_elbow", [0.0, -0.35, 0.0])
    set_joint_pose("right_elbow", [0.0, 0.35, 0.0])

    # 下肢：轻微走路姿态
    set_joint_pose("left_hip", [0.25, 0.0, 0.08])
    set_joint_pose("right_hip", [-0.18, 0.0, -0.08])
    set_joint_pose("left_knee", [0.35, 0.0, 0.0])
    set_joint_pose("right_knee", [0.20, 0.0, 0.0])

    # 躯干轻微旋转，增强 3D 感
    set_joint_pose("spine1", [0.0, 0.0, 0.08])
    set_joint_pose("spine2", [0.0, 0.0, -0.05])
    set_joint_pose("spine3", [0.0, 0.0, 0.04])

    return global_orient, body_pose


# ============================================================
# 4. 手写 LBS
# ============================================================

def prepare_posedirs(posedirs: torch.Tensor, expected_pose_dim: int):
    """
    不同版本里 posedirs 的形状可能是 [P, V*3]，也可能是 [V*3, P]。
    官方 LBS 需要 [P, V*3]。
    """
    if posedirs.dim() != 2:
        posedirs = posedirs.reshape(posedirs.shape[0], -1)

    if posedirs.shape[0] == expected_pose_dim:
        return posedirs
    if posedirs.shape[1] == expected_pose_dim:
        return posedirs.T

    raise RuntimeError(
        f"posedirs 形状与 pose_feature 不匹配，posedirs.shape={tuple(posedirs.shape)}, "
        f"expected_pose_dim={expected_pose_dim}"
    )


def compute_manual_lbs(model, betas, global_orient, body_pose):
    device = betas.device
    dtype = betas.dtype

    v_template = model.v_template
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)  # [1, V, 3]

    shapedirs = model.shapedirs[:, :, :betas.shape[1]]

    # (b) 形状混合：v_shaped = v_template + B_S(beta)
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 由形状后的网格回归关节：J(beta)
    J = vertices2joints(model.J_regressor, v_shaped)

    # (c) 姿态校正：v_posed = v_shaped + B_P(theta)
    full_pose = torch.cat([global_orient, body_pose], dim=1)  # [1, 24*3]
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)

    posedirs = prepare_posedirs(model.posedirs, expected_pose_dim=pose_feature.shape[1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets

    # (d) 刚体层级变换 + LBS
    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)

    num_joints = J.shape[1]
    W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)  # [1, V, J]

    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

    homogen_coord = torch.ones((1, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)  # [1, V, 4]
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))       # [1, V, 4, 1]
    verts = v_homo[:, :, :3, 0]

    # 模板姿态下的关节，方便可视化阶段 (a)
    J_template = vertices2joints(model.J_regressor, v_template)

    return {
        "v_template": v_template,
        "J_template": J_template,
        "v_shaped": v_shaped,
        "J_shaped": J,
        "pose_offsets": pose_offsets,
        "v_posed": v_posed,
        "J_transformed": J_transformed,
        "verts": verts,
    }


def compare_with_official_forward(model, betas, global_orient, body_pose, manual_verts):
    with torch.no_grad():
        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True,
        )
    official_verts = output.vertices
    diff = torch.abs(manual_verts - official_verts)
    mean_err = diff.mean().item()
    max_err = diff.max().item()
    return mean_err, max_err


# ============================================================
# 5. 主程序
# ============================================================

def main(args):
    device = torch.device("cpu")
    dtype = torch.float32

    # 兼容两套参数名
    model_arg = args.model_path if args.model_path is not None else args.model_dir
    out_arg = args.out if args.out is not None else args.out_dir

    model_dir = normalize_model_path_for_smplx(model_arg)
    out_dir = resolve_script_path(out_arg)
    make_out_dir(out_dir)

    print("========== 加载 SMPL ==========")
    print("model_dir =", model_dir)
    print("out_dir   =", out_dir)

    install_chumpy_pickle_shim()
    model = smplx.create(
        model_path=model_dir,
        model_type="smpl",
        gender=args.gender,
        ext="pkl",
        num_betas=args.num_betas,
        batch_size=1,
    ).to(device)
    model.eval()

    faces = np.asarray(model.faces, dtype=np.int32)
    parents = to_numpy(model.parents).astype(np.int64)

    num_vertices = int(model.v_template.shape[0])
    num_faces = int(faces.shape[0])
    num_joints = int(model.lbs_weights.shape[1])

    print(f"顶点数: {num_vertices}")
    print(f"面片数: {num_faces}")
    print(f"关节数: {num_joints}")
    print(f"betas 维度: {args.num_betas}")

    print("\n========== 构造 beta 和 pose ==========")
    betas = build_demo_shape(device, dtype, num_betas=args.num_betas)
    global_orient, body_pose = build_demo_pose(device, dtype)

    print("betas =", to_numpy(betas))
    print("body_pose 非零元素数 =", int((body_pose.abs() > 1e-8).sum().item()))

    print("\n========== 手写 LBS ==========")
    with torch.no_grad():
        data = compute_manual_lbs(model, betas, global_orient, body_pose)

    print("v_template:", tuple(data["v_template"].shape))
    print("v_shaped:  ", tuple(data["v_shaped"].shape))
    print("J_shaped:  ", tuple(data["J_shaped"].shape))
    print("v_posed:   ", tuple(data["v_posed"].shape))
    print("verts:     ", tuple(data["verts"].shape))

    print("\n========== 与官方 forward 对比 ==========")
    try:
        mean_err, max_err = compare_with_official_forward(
            model, betas, global_orient, body_pose, data["verts"]
        )
        print(f"手写 LBS 与官方 forward 的平均绝对误差: {mean_err:.10e}")
        print(f"手写 LBS 与官方 forward 的最大绝对误差: {max_err:.10e}")
    except Exception as e:
        mean_err, max_err = float("nan"), float("nan")
        print("[警告] 官方 forward 对比失败，但图片仍会生成。原因：")
        print(str(e))

    joint_id = int(args.joint_id)
    if joint_id < 0 or joint_id >= model.lbs_weights.shape[1]:
        raise ValueError(
            f"joint_id 越界：{joint_id}，可选范围应为 [0, {model.lbs_weights.shape[1] - 1}]"
        )

    weight_scalar = to_numpy(model.lbs_weights[:, joint_id])
    pose_offset_norm = np.linalg.norm(to_numpy(data["pose_offsets"][0]), axis=1)

    v_template_np = to_numpy(data["v_template"][0])
    J_template_np = to_numpy(data["J_template"][0])
    v_shaped_np = to_numpy(data["v_shaped"][0])
    J_shaped_np = to_numpy(data["J_shaped"][0])
    v_posed_np = to_numpy(data["v_posed"][0])
    verts_np = to_numpy(data["verts"][0])
    J_transformed_np = to_numpy(data["J_transformed"][0])
    lbs_weights_np = to_numpy(model.lbs_weights)

    print("\n========== 生成 3D 可视化图片 ==========")

    save_single_figure(
        os.path.join(out_dir, "stage_a_template_weights.png"),
        v_template_np,
        faces,
        joints=J_template_np,
        parents=parents,
        vertex_scalar=weight_scalar,
        title=f"(a) Template Mesh + Weight of Joint {joint_id}",
    )
    print("[保存] stage_a_template_weights.png")

    save_single_figure(
        os.path.join(out_dir, "stage_b_shaped_joints.png"),
        v_shaped_np,
        faces,
        joints=J_shaped_np,
        parents=parents,
        vertex_scalar=None,
        title="(b) Shape Blend + Joint Regression",
    )
    print("[保存] stage_b_shaped_joints.png")

    save_single_figure(
        os.path.join(out_dir, "stage_c_pose_offsets.png"),
        v_posed_np,
        faces,
        joints=J_shaped_np,
        parents=parents,
        vertex_scalar=pose_offset_norm,
        title="(c) Pose Blend Shapes (colored by |pose_offsets|)",
    )
    print("[保存] stage_c_pose_offsets.png")

    save_single_figure(
        os.path.join(out_dir, "stage_d_lbs_result.png"),
        verts_np,
        faces,
        joints=J_transformed_np,
        parents=parents,
        vertex_scalar=None,
        title="(d) Final LBS Result",
    )
    print("[保存] stage_d_lbs_result.png")

    grid_dict = {
        "v_template": v_template_np,
        "J_template": J_template_np,
        "v_shaped": v_shaped_np,
        "J_shaped": J_shaped_np,
        "v_posed": v_posed_np,
        "verts": verts_np,
        "J_transformed": J_transformed_np,
        "weight_scalar": weight_scalar,
        "pose_offset_norm": pose_offset_norm,
    }
    save_comparison_grid(
        os.path.join(out_dir, "comparison_grid.png"),
        grid_dict,
        faces,
        parents=parents,
    )
    print("[保存] comparison_grid.png")

    save_all_joint_weights_figure(
        os.path.join(out_dir, "all_joint_weights.png"),
        v_template_np,
        faces,
        J_template_np,
        parents,
        lbs_weights_np,
    )
    print("[保存] all_joint_weights.png")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== SMPL LBS Lab Summary =====\n")
        f.write(f"model_dir: {model_dir}\n")
        f.write(f"num_vertices: {num_vertices}\n")
        f.write(f"num_faces: {num_faces}\n")
        f.write(f"num_joints(from lbs_weights): {num_joints}\n")
        f.write(f"num_betas: {args.num_betas}\n")
        f.write(f"visualized_joint_id: {joint_id}\n")
        f.write(f"v_template shape: {tuple(data['v_template'].shape)}\n")
        f.write(f"v_shaped shape: {tuple(data['v_shaped'].shape)}\n")
        f.write(f"J_shaped shape: {tuple(data['J_shaped'].shape)}\n")
        f.write(f"v_posed shape: {tuple(data['v_posed'].shape)}\n")
        f.write(f"verts shape: {tuple(data['verts'].shape)}\n")
        f.write(f"manual_vs_official_mean_abs_error: {mean_err:.10e}\n")
        f.write(f"manual_vs_official_max_abs_error: {max_err:.10e}\n")
        f.write("\nOutput files:\n")
        f.write("stage_a_template_weights.png\n")
        f.write("all_joint_weights.png\n")
        f.write("stage_b_shaped_joints.png\n")
        f.write("stage_c_pose_offsets.png\n")
        f.write("stage_d_lbs_result.png\n")
        f.write("comparison_grid.png\n")
        f.write("summary.txt\n")
    print("[保存] summary.txt")

    print("\n========== 运行完成 ==========")
    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 兼容你之前的参数名
    parser.add_argument("--model-path", type=str, default=None, help="模型路径，推荐写 models")
    parser.add_argument("--out", type=str, default=None, help="输出目录，推荐写 outputs")

    # 兼容老师参考代码的参数名
    parser.add_argument("--model-dir", type=str, default="./models", help="模型目录，内部应包含 smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="输出目录")

    parser.add_argument("--joint-id", type=int, default=18, help="要可视化权重的关节编号，默认 18")
    parser.add_argument("--num-betas", type=int, default=10, help="使用多少个 shape 参数，默认 10")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"], help="SMPL 性别，默认 neutral")

    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        print("\n程序运行失败，完整错误如下：")
        traceback.print_exc()
        print(
            "\n常见检查项：\n"
            "1. 模型文件是否存在：models/smpl/SMPL_NEUTRAL.pkl\n"
            "2. PyCharm Parameters 是否为：--model-path models --out outputs\n"
            "3. Working directory 是否为项目根目录\n"
            "4. 当前环境是否有 torch、smplx、numpy、matplotlib\n"
        )
        raise
