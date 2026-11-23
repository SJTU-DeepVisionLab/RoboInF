import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    (x, y, z) = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def origin_matrix(element, scale):
    matrix = np.eye(4)
    if element is not None:
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ") * scale
        (r, p, y) = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        matrix[:3, :3] = (
            rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
        )
    return matrix


def inspect_asset(root, model_id=1, contact_point_id=0):
    directory = Path(root) / "assets/objects/056_switch"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing switch assets: {directory}")
    models = sorted(
        (p for p in directory.iterdir() if p.is_dir() and p.name != "visual"),
        key=lambda p: int(re.search("\\d+", p.name).group()),
    )
    model = models[model_id] if model_id < len(models) else directory / str(model_id)
    metadata = json.loads((model / "model_data.json").read_text())
    urdf = ET.parse(model / "mobility.urdf").getroot()
    scale = float(metadata["scale"])
    if "transform_matrix" in metadata and (
        not np.allclose(metadata["transform_matrix"], np.eye(4))
    ):
        raise ValueError(
            "Switch assets with extra loader transforms need a calibrated mount"
        )
    active = [joint for joint in urdf.findall("joint") if joint.get("type") != "fixed"]
    if len(active) != 1 or active[0].get("type") != "revolute":
        raise ValueError("Three-switch task requires one revolute joint per switch")
    joint = active[0]
    limits = [float(joint.find("limit").get(key)) for key in ("lower", "upper")]
    if not np.isfinite(limits).all() or limits[1] <= limits[0]:
        raise ValueError("Invalid switch joint limits")
    contact = metadata["contact_points"][contact_point_id]
    links = {link.get("name"): link for link in urdf.findall("link")}
    if contact["base"] not in links:
        raise ValueError("Contact point references a missing link")
    vertices = {}
    for link in links.values():
        parts = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected OBJ mesh collision geometry for switch")
            path = model / mesh.get("filename")
            points = np.array(
                [
                    [float(v) for v in line.split()[1:4]]
                    for line in path.read_text().splitlines()
                    if line.startswith("v ")
                ]
            )
            points *= scale * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            transform = origin_matrix(collision.find("origin"), scale)
            parts.append(points @ transform[:3, :3].T + transform[:3, 3])
        vertices[link.get("name")] = (
            np.concatenate(parts) if parts else np.empty((0, 3))
        )
    children = {j.find("child").get("link") for j in urdf.findall("joint")}
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError("Expected one articulation root")
    endpoints = []
    root_name = next(iter(roots))
    for angle in np.linspace(*limits, 17):
        transforms = {root_name: np.eye(4)}
        pending = list(urdf.findall("joint"))
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in transforms]
            if not ready:
                raise ValueError("Cyclic or disconnected URDF")
            for j in ready:
                transform = origin_matrix(j.find("origin"), scale)
                if j.get("type") != "fixed":
                    transform[:3, :3] = transform[:3, :3] @ rotation(
                        np.fromstring(j.find("axis").get("xyz"), sep=" "), angle
                    )
                transforms[j.find("child").get("link")] = (
                    transforms[j.find("parent").get("link")] @ transform
                )
                pending.remove(j)
        points = np.concatenate(
            [
                points @ transforms[name][:3, :3].T + transforms[name][:3, 3]
                for (name, points) in vertices.items()
            ]
        )
        endpoints.append(points)
    points = np.concatenate(endpoints)
    return {
        "scale": scale,
        "root_bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
    }
