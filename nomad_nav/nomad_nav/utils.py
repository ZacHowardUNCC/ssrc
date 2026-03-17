import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from typing import List
from sensor_msgs.msg import Image

from nomad_nav.path_utils import ensure_visualnav_python_paths

ensure_visualnav_python_paths()

from vint_train.data.data_utils import IMAGE_ASPECT_RATIO


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)."""
    model_type = config["model_type"]

    if model_type == "gnm":
        from vint_train.models.gnm.gnm import GNM

        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif model_type == "vint":
        from vint_train.models.vint.vint import ViNT

        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif model_type == "nomad":
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            from vint_train.models.vint.vit import ViT

            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    """Convert a ROS2 sensor_msgs/Image to a PIL Image.

    Handles common encodings: rgb8, bgr8, rgba8, bgra8, mono8, yuyv, uyvy.
    Uses msg.step when present to safely handle row padding.
    """
    encoding = (msg.encoding.lower() if msg.encoding else "rgb8").replace("-", "_")
    raw = bytes(msg.data)
    h = msg.height
    w = msg.width
    step = int(msg.step) if getattr(msg, "step", 0) else 0

    def rows_2d(bytes_per_row: int) -> np.ndarray:
        flat = np.frombuffer(raw, dtype=np.uint8)
        expected = h * bytes_per_row
        if flat.size < expected:
            raise ValueError(
                f"Image buffer too small: have {flat.size} bytes, need {expected}"
            )
        return flat[:expected].reshape(h, bytes_per_row)

    def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        y = y.astype(np.float32)
        u = u.astype(np.float32) - 128.0
        v = v.astype(np.float32) - 128.0
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def yuyv_to_rgb(step_bytes: int) -> PILImage.Image:
        if w % 2 != 0:
            raise ValueError("YUYV decoding requires even image width")
        row = rows_2d(step_bytes)
        packed = row[:, : w * 2].reshape(h, w // 2, 4)
        y0 = packed[:, :, 0]
        u = packed[:, :, 1]
        y1 = packed[:, :, 2]
        v = packed[:, :, 3]
        y = np.empty((h, w), dtype=np.uint8)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        u_full = np.repeat(u, 2, axis=1)
        v_full = np.repeat(v, 2, axis=1)
        return PILImage.fromarray(yuv_to_rgb(y, u_full, v_full), "RGB")

    def uyvy_to_rgb(step_bytes: int) -> PILImage.Image:
        if w % 2 != 0:
            raise ValueError("UYVY decoding requires even image width")
        row = rows_2d(step_bytes)
        packed = row[:, : w * 2].reshape(h, w // 2, 4)
        u = packed[:, :, 0]
        y0 = packed[:, :, 1]
        v = packed[:, :, 2]
        y1 = packed[:, :, 3]
        y = np.empty((h, w), dtype=np.uint8)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        u_full = np.repeat(u, 2, axis=1)
        v_full = np.repeat(v, 2, axis=1)
        return PILImage.fromarray(yuv_to_rgb(y, u_full, v_full), "RGB")

    if encoding in ("rgb8",):
        row = rows_2d(step or (w * 3))
        img = row[:, : w * 3].reshape(h, w, 3)
        return PILImage.fromarray(img, "RGB")
    elif encoding in ("bgr8",):
        row = rows_2d(step or (w * 3))
        img = row[:, : w * 3].reshape(h, w, 3)
        return PILImage.fromarray(img[:, :, ::-1].copy(), "RGB")
    elif encoding in ("rgba8",):
        row = rows_2d(step or (w * 4))
        img = row[:, : w * 4].reshape(h, w, 4)
        return PILImage.fromarray(img[:, :, :3], "RGB")
    elif encoding in ("bgra8",):
        row = rows_2d(step or (w * 4))
        img = row[:, : w * 4].reshape(h, w, 4)
        return PILImage.fromarray(img[:, :, 2::-1].copy(), "RGB")
    elif encoding in ("mono8",):
        row = rows_2d(step or w)
        img = row[:, :w]
        return PILImage.fromarray(img, "L").convert("RGB")
    elif encoding in ("yuyv", "yuy2", "yuv422", "yuv422_yuy2", "yuv422_yuyv"):
        return yuyv_to_rgb(step or (w * 2))
    elif encoding in ("uyvy", "yuv422_uyvy"):
        return uyvy_to_rgb(step or (w * 2))
    else:
        # Best-effort fallback: try RGB layout, then grayscale.
        if step >= w * 3:
            row = rows_2d(step)
            img = row[:, : w * 3].reshape(h, w, 3)
            return PILImage.fromarray(img, "RGB")
        row = rows_2d(step or w)
        img = row[:, :w]
        return PILImage.fromarray(img, "L").convert("RGB")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def transform_images(
    pil_imgs: List[PILImage.Image],
    image_size: List[int],
    center_crop: bool = False,
) -> torch.Tensor:
    """Transform a list of PIL images (or a single image) to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def clip_angle(angle: float) -> float:
    """Clip angle to [-pi, pi]."""
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
