"""Published DExplore Inspire observation contract (2 x 721 = 1442)."""
from __future__ import annotations

import torch


REFERENCE_DIM = 428
KEY_BODY_COUNT = 16
IG_KEY_INDICES = (0, 3, 6, 9, 12, 15)
CONTACT_KEY_INDICES = (3, 6, 9, 12, 15)


def select_contact_forces(net_contact_force: torch.Tensor, query_indices,
                          contact_indices) -> torch.Tensor:
    """Select contact bodies without reinterpreting velocity as force."""
    query = torch.as_tensor(query_indices, dtype=torch.long, device=net_contact_force.device)
    contact = torch.as_tensor(contact_indices, dtype=torch.long, device=net_contact_force.device)
    return net_contact_force.index_select(1, query).index_select(1, contact)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError(f"Quaternion shapes must match [...,4], got {a.shape} and {b.shape}")
    shape = a.shape
    a, b = a.reshape(-1, 4), b.reshape(-1, 4)
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    result = torch.stack((
        qq - xx + (x1 + w1) * (x2 + w2),
        qq - yy + (w1 - x1) * (y2 + z2),
        qq - zz + (z1 + y1) * (w2 - x2),
        qq - ww + (z1 - y1) * (y2 - z2),
    ), dim=-1)
    return result.view(shape)


def quat_rotate(q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if q.shape[:-1] != value.shape[:-1] or q.shape[-1] != 4 or value.shape[-1] != 3:
        raise ValueError(f"Expected matching quaternion/vector batches, got {q.shape} and {value.shape}")
    shape = value.shape
    q, value = q.reshape(-1, 4), value.reshape(-1, 3)
    q_w, q_vec = q[:, -1], q[:, :3]
    a = value * (2.0 * q_w.square() - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, value, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(-1, 1, 3), value.view(-1, 3, 1)).squeeze(-1) * 2.0
    return (a + b + c).view(shape)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    q = torch.where(q[..., 3:] < 0, -q, q)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_mul_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return quat_normalize(quat_mul(a, b))


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = (angle / 2.0).unsqueeze(-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return quat_normalize(torch.cat((axis * half.sin(), half.cos()), dim=-1))


def normalize_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def exp_map_to_quat(exp_map: torch.Tensor) -> torch.Tensor:
    angle = exp_map.norm(dim=-1)
    axis = exp_map / angle.unsqueeze(-1).clamp_min(1e-9)
    angle = normalize_angle(angle)
    valid = angle.abs() > 1e-5
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1
    axis = torch.where(valid.unsqueeze(-1), axis, default_axis)
    angle = torch.where(valid, angle, torch.zeros_like(angle))
    return quat_from_angle_axis(angle, axis)


def quat_to_exp_map(q: torch.Tensor) -> torch.Tensor:
    # Preserve the released helper literally: source quaternions are assumed
    # normalized and are not silently renormalized before finite differencing.
    sine = torch.sqrt(1.0 - q[..., 3].square())
    angle = normalize_angle(2.0 * torch.acos(q[..., 3]))
    axis = q[..., :3] / sine.unsqueeze(-1)
    valid = sine.abs() > 1e-5
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1
    axis = torch.where(valid.unsqueeze(-1), axis, default_axis)
    angle = torch.where(valid, angle, torch.zeros_like(angle))
    return angle.unsqueeze(-1) * axis


def calc_heading(q: torch.Tensor) -> torch.Tensor:
    reference = torch.zeros_like(q[..., :3])
    reference[..., 0] = 1
    direction = quat_rotate(q, reference)
    return torch.atan2(direction[..., 1], direction[..., 0])


def calc_heading_quat(q: torch.Tensor) -> torch.Tensor:
    axis = torch.zeros_like(q[..., :3])
    axis[..., 2] = 1
    return quat_from_angle_axis(calc_heading(q), axis)


def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    axis = torch.zeros_like(q[..., :3])
    axis[..., 2] = 1
    return quat_from_angle_axis(-calc_heading(q), axis)


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros_like(q[..., :3])
    tangent[..., 0] = 1
    normal = torch.zeros_like(q[..., :3])
    normal[..., -1] = 1
    return torch.cat((quat_rotate(q, tangent), quat_rotate(q, normal)), dim=-1)


def compute_sdf(points: torch.Tensor, object_points: torch.Tensor) -> torch.Tensor:
    """Match DExplore's closest sampled-surface displacement vector."""
    differences = points.unsqueeze(2) - object_points.unsqueeze(1)
    closest = differences.norm(dim=-1).argmin(dim=-1)
    batch = torch.arange(points.shape[0], device=points.device).unsqueeze(1)
    point = torch.arange(points.shape[1], device=points.device).unsqueeze(0)
    return differences[batch, point, closest].contiguous()


def object_points_world(object_state: torch.Tensor, object_points: torch.Tensor) -> torch.Tensor:
    batch = object_state.shape[0]
    if object_points.ndim == 2:
        object_points = object_points.unsqueeze(0).expand(batch, -1, -1)
    if object_points.shape[0] != batch or object_points.shape[-1] != 3:
        raise ValueError(f"Invalid object point shape {tuple(object_points.shape)}")
    rotation = object_state[:, 3:7].unsqueeze(1).expand(-1, object_points.shape[1], -1)
    return quat_rotate(rotation, object_points) + object_state[:, None, :3]


def _body_observation(body_pos: torch.Tensor, body_rot: torch.Tensor,
                      body_vel: torch.Tensor, body_ang_vel: torch.Tensor,
                      contact_forces: torch.Tensor, ref_obs: torch.Tensor,
                      key_body_ids: torch.Tensor, contact_body_ids: torch.Tensor,
                      root_body_id: int, tracking_root_body_id: int) -> torch.Tensor:
    root_pos = body_pos[:, root_body_id]
    root_rot = body_rot[:, root_body_id]
    heading = calc_heading_quat_inv(root_rot)
    heading_inv = calc_heading_quat(root_rot)
    count = int(key_body_ids.numel())
    flat_heading = heading[:, None].expand(-1, count, -1).reshape(-1, 4)
    flat_heading_inv = heading_inv[:, None].expand(-1, count, -1).reshape(-1, 4)

    ref_body_pos = ref_obs[:, 119:119 + count * 3].view(-1, count, 3)
    current_body_pos = body_pos.index_select(1, key_body_ids)
    diff_body_pos = quat_rotate(
        flat_heading, (ref_body_pos - current_body_pos).reshape(-1, 3)).view(-1, count * 3)

    # This duplicated-current block is a legacy checkpoint contract despite its
    # original local_ref_body_pos name.
    local_ref_body_pos = quat_rotate(
        flat_heading, (current_body_pos - root_pos[:, None]).reshape(-1, 3)).view(-1, count * 3)
    local_body_pos = local_ref_body_pos[..., 3:]

    current_body_rot = body_rot.index_select(1, key_body_ids)
    local_body_rot = quat_mul(flat_heading, current_body_rot.reshape(-1, 4))
    local_body_rot_obs = quat_to_tan_norm(local_body_rot).view(-1, count * 6)

    ref_rot_start = 119 + count * 3 + 1 + 16 + count * 3
    ref_body_rot = ref_obs[:, ref_rot_start:ref_rot_start + count * 4].view(-1, count, 4)
    diff_global_rot = quat_mul_norm(
        quat_inverse(ref_body_rot.reshape(-1, 4)), current_body_rot.reshape(-1, 4))
    diff_local_rot = quat_mul(quat_mul(flat_heading, diff_global_rot), flat_heading_inv)
    diff_local_rot_obs = quat_to_tan_norm(diff_local_rot).view(-1, count * 6)
    local_ref_body_rot = quat_to_tan_norm(
        quat_mul(flat_heading, ref_body_rot.reshape(-1, 4))).view(-1, count * 6)

    ref_vel_start = ref_rot_start + count * 4
    ref_body_vel = ref_obs[:, ref_vel_start:ref_vel_start + count * 3].view(-1, count, 3)
    current_body_vel = body_vel.index_select(1, key_body_ids)
    diff_local_vel = quat_rotate(
        flat_heading, (ref_body_vel - current_body_vel).reshape(-1, 3)).view(-1, count * 3)

    ref_ang_start = ref_vel_start + count * 3
    ref_body_ang = ref_obs[:, ref_ang_start:ref_ang_start + count * 3].view(-1, count, 3)
    current_body_ang = body_ang_vel.index_select(1, key_body_ids)
    diff_local_ang = quat_rotate(
        flat_heading, (ref_body_ang - current_body_ang).reshape(-1, 3)).view(-1, count * 3)

    local_body_vel = quat_rotate(flat_heading, current_body_vel.reshape(-1, 3)).view(-1, count * 3)
    local_body_ang = quat_rotate(flat_heading, current_body_ang.reshape(-1, 3)).view(-1, count * 3)

    contact = (contact_forces.index_select(1, contact_body_ids).abs().amax(dim=-1) > 0.1).to(body_pos.dtype)
    ref_contact = ref_obs[:, 119 + count * 3 + 1:119 + count * 3 + 1 + 16]
    ref_contact = ref_contact[:, list(CONTACT_KEY_INDICES)]
    diff_contact = ref_contact * ((ref_contact + 1.0) / 2.0 - contact)

    tracking_pos = body_pos[:, tracking_root_body_id]
    tracking_rot = body_rot[:, tracking_root_body_id]
    diff_root_rot = quat_mul_norm(quat_inverse(ref_obs[:, :4]), tracking_rot)
    result = torch.cat((
        ref_obs[:, 4:7] - tracking_pos,
        quat_to_tan_norm(diff_root_rot),
        body_vel[:, tracking_root_body_id],
        body_ang_vel[:, tracking_root_body_id],
        local_body_pos,
        local_body_rot_obs,
        local_body_vel,
        local_body_ang,
        contact,
        diff_body_pos,
        diff_local_rot_obs,
        diff_contact,
        local_ref_body_pos,
        local_ref_body_rot,
        diff_local_vel,
        diff_local_ang,
    ), dim=-1)
    if result.shape[-1] != 646:
        raise RuntimeError(f"DExplore body observation width mismatch: {result.shape[-1]}")
    return result


def _object_observation(body_pos: torch.Tensor, body_rot: torch.Tensor,
                        object_state: torch.Tensor, ref_obs: torch.Tensor,
                        root_body_id: int) -> torch.Tensor:
    root_pos = body_pos[:, root_body_id]
    root_rot = body_rot[:, root_body_id]
    heading = calc_heading_quat_inv(root_rot)
    heading_inv = calc_heading_quat(root_rot)
    object_pos, object_rot = object_state[:, :3], object_state[:, 3:7]
    object_vel, object_ang = object_state[:, 7:10], object_state[:, 10:13]

    local_object_pos = quat_rotate(heading, object_pos - root_pos)
    local_object_rot = quat_to_tan_norm(quat_mul(heading, object_rot))
    local_object_vel = quat_rotate(heading, object_vel)
    local_object_ang = quat_rotate(heading, object_ang)

    ref_object_pos = ref_obs[:, 106:109]
    diff_object_pos = quat_rotate(heading, ref_object_pos - object_pos)
    local_ref_object_pos = quat_rotate(heading, ref_object_pos - ref_obs[:, 4:7])
    ref_object_rot = ref_obs[:, 109:113]
    diff_global_rot = quat_mul_norm(quat_inverse(ref_object_rot), object_rot)
    diff_local_rot = quat_mul(quat_mul(heading, diff_global_rot), heading_inv)
    diff_object_rot = quat_to_tan_norm(diff_local_rot)
    local_ref_object_rot = quat_to_tan_norm(quat_mul(heading, ref_object_rot))
    diff_object_vel = quat_rotate(heading, ref_obs[:, 113:116] - object_vel)
    diff_object_ang = quat_rotate(heading, ref_obs[:, 116:119] - object_ang)

    result = torch.cat((
        local_object_pos, local_object_rot, local_object_vel, local_object_ang,
        diff_object_pos, diff_object_rot, local_ref_object_pos,
        local_ref_object_rot, diff_object_vel, diff_object_ang,
    ), dim=-1)
    if result.shape[-1] != 39:
        raise RuntimeError(f"DExplore object observation width mismatch: {result.shape[-1]}")
    return result


def build_dexplore_observation(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    body_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
    contact_forces: torch.Tensor,
    object_state: torch.Tensor,
    ref_obs: torch.Tensor,
    object_points: torch.Tensor,
    key_body_ids,
    contact_body_ids,
    *,
    root_body_id: int = 7,
    tracking_root_body_id: int = 6,
) -> torch.Tensor:
    """Build one 721D observation with the released DExplore field semantics."""
    batch = body_pos.shape[0]
    expected_body = (batch, body_pos.shape[1])
    for name, value, tail in (
        ("body_rot", body_rot, 4), ("body_vel", body_vel, 3),
        ("body_ang_vel", body_ang_vel, 3), ("contact_forces", contact_forces, 3),
    ):
        if value.shape[:2] != expected_body or value.shape[-1] != tail:
            raise ValueError(f"{name} shape {tuple(value.shape)} is incompatible with body_pos {tuple(body_pos.shape)}")
    if ref_obs.shape != (batch, REFERENCE_DIM) or object_state.shape != (batch, 13):
        raise ValueError(f"Invalid reference/object shapes: {tuple(ref_obs.shape)}, {tuple(object_state.shape)}")
    key_body_ids = torch.as_tensor(key_body_ids, dtype=torch.long, device=body_pos.device)
    contact_body_ids = torch.as_tensor(contact_body_ids, dtype=torch.long, device=body_pos.device)
    if key_body_ids.numel() != KEY_BODY_COUNT or contact_body_ids.numel() != 5:
        raise ValueError("DExplore requires 16 key bodies and 5 contact bodies")

    body = _body_observation(
        body_pos, body_rot, body_vel, body_ang_vel, contact_forces,
        ref_obs, key_body_ids, contact_body_ids, root_body_id, tracking_root_body_id)
    object_obs = _object_observation(body_pos, body_rot, object_state, ref_obs, root_body_id)

    world_points = object_points_world(object_state, object_points)
    ig_ids = key_body_ids[torch.as_tensor(IG_KEY_INDICES, device=body_pos.device)]
    actual_key_pos = body_pos.index_select(1, ig_ids)
    ig = compute_sdf(actual_key_pos, world_points)
    heading = calc_heading_quat_inv(body_rot[:, root_body_id])
    ig = quat_rotate(heading[:, None].expand(-1, len(IG_KEY_INDICES), -1), ig).reshape(batch, -1)
    ref_ig = ref_obs[:, 184:184 + KEY_BODY_COUNT * 3].view(batch, KEY_BODY_COUNT, 3)
    ref_ig = ref_ig[:, list(IG_KEY_INDICES)].reshape(batch, -1)
    result = torch.cat((body, object_obs, ig, ref_ig - ig), dim=-1)
    if result.shape[-1] != 721 or not torch.isfinite(result).all():
        raise RuntimeError("Invalid DExplore 721D observation")
    return result
