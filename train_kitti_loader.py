"""
KITTI Dataset & DataLoader for SFPR training/validation.

Returns per sample:
  - LiDAR query input (point cloud in polar BEV)
  - OSM candidate map (rasterized tile)
  - Semantic BEV from Cylinder3D predictions
  - PC visibility mask (max range per azimuth)
  - OSM structural anchor candidates (fingerprint element candidates)
  - PC structural anchor candidates (fingerprint element candidates)
  - Augmentation parameters (for spatial alignment of saliency fields)
  - Dataset indices

Structural anchor types (five categories, §4.1):
  road_fork, building_corner, trunk, pole, traffic_sign

Unified config root:
    loading:
      kitti:
        pc_data_path: "/home/data/neurips/DataSet/KITTI/dataset/"
        label_data_path: "/home/data/neurips/DataSet/KITTI/dataset/"
        node_path: "data/kitti/nodes"
        osm_data_path: "data/kitti/osm"
        pose_path: "data/kitti/pose"
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import numba as nb
import pykitti
import torch
import torch.utils.data as data
import yaml

from maploc.osm.tiling import TileManager, BoundaryBox
from maploc.utils.geo import Projection

logger = logging.getLogger(__name__)

def _resolve_dataset_split(opt: dict, mode: str):
    candidates = [
        opt.get('loading', {}).get('dataset_split'),
        opt.get('dataset_split'),
        opt.get('loading', {}).get('kitti', {}).get('dataset_split'),
    ]
    for c in candidates:
        if isinstance(c, dict):
            if mode == 'train' and 'train' in c:
                return c['train']
            if mode in ('val', 'eval') and 'val' in c:
                return c['val']
    return []


def _zseq(seq: str) -> str:
    return str(seq).zfill(2)


def _get_kitti_cfg(opt: dict, seq: str, split: str) -> dict:
    root = opt.get('loading', {}).get('kitti')
    if not isinstance(root, dict) or len(root) == 0:
        raise KeyError('Missing config loading.kitti with keys: pc_data_path, label_data_path, node_path, osm_data_path, pose_path')
    required = ['pc_data_path', 'label_data_path', 'node_path', 'osm_data_path', 'pose_path']
    missing = [k for k in required if k not in root]
    if missing:
        raise KeyError(f'loading.kitti missing required keys: {missing}')
    seq = _zseq(seq)
    split_tag = 'Train' if split.lower() == 'train' else 'Test'
    return {
        'seq': seq,
        'pc_data_root': str(root['pc_data_path']),
        'pc_bin_dir': os.path.join(str(root['pc_data_path']), 'sequences', seq, 'velodyne'),
        'pc_label_dir': os.path.join(str(root['label_data_path']), 'sequences', seq, 'label_cylinder'),
        'node_root': str(root['node_path']),
        'osm_node_pkl_path': os.path.join(str(root['node_path']), f'KITTI_OSM_{seq}_{split_tag}.pkl'),
        'pc_node_pkl_path': os.path.join(str(root['node_path']), f'KITTI_PC_{seq}_{split_tag}.pkl'),
        'tile_pkl_path': os.path.join(str(root['osm_data_path']), f'100_Tiles_{seq}_{split_tag}.pkl'),
        'osm_file_path': os.path.join(str(root['osm_data_path']), 'Karlsruhe.osm'),
        'pose_npy_path': os.path.join(str(root['pose_path']), f'gps_sequence_{seq}.npy'),
    }


class NodeLoader:
    TARGET_KEYS = ['road_fork', 'building_corner', 'trunk', 'pole', 'traffic_sign']

    def __init__(self, pkl_path_resolver):
        self._resolver = pkl_path_resolver
        self._cache: Dict[str, Dict[int, Dict]] = {}

    def load_sequence(self, seq: str) -> None:
        seq = _zseq(seq)
        if seq in self._cache:
            return
        pkl_path = Path(self._resolver(seq))
        if not pkl_path.exists():
            raise FileNotFoundError(f'Required node PKL not found: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        self._cache[seq] = {}
        frames = raw_data.get('frames', []) if isinstance(raw_data, dict) else raw_data
        for fi, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            try:
                idx = int(frame.get('bin_name', '0'))
            except (ValueError, TypeError):
                idx = int((frame.get('metadata', {}) or {}).get('frame_idx', fi))
            self._cache[seq][idx] = frame

    def get_nodes_raw(self, seq: str, frame_idx: int) -> Dict:
        seq = _zseq(seq)
        if seq not in self._cache:
            self.load_sequence(seq)
        frame_data = self._cache[seq].get(frame_idx)
        return frame_data.get('nodes', {}) if frame_data else {}


@nb.jit(nopython=True, cache=True)
def create_semantic_bev_numba(r_idx: np.ndarray, phi_idx: np.ndarray, labels: np.ndarray, R_bins: int, Phi_bins: int) -> np.ndarray:
    count_bev = np.zeros((R_bins, Phi_bins, 20), dtype=np.int32)
    for i in range(len(r_idx)):
        r = r_idx[i]
        p = phi_idx[i]
        l = labels[i]
        if 0 <= r < R_bins and 0 <= p < Phi_bins and 0 <= l < 20:
            count_bev[r, p, l] += 1
    semantic_bev = np.zeros((R_bins, Phi_bins), dtype=np.int64)
    for i in range(R_bins):
        for j in range(Phi_bins):
            max_count = 0
            max_label = 0
            for k in range(20):
                if count_bev[i, j, k] > max_count:
                    max_count = count_bev[i, j, k]
                    max_label = k
            semantic_bev[i, j] = max_label
    return semantic_bev


def compute_distance_feature_polar(xyz_pol, r_bins=480, phi_bins=360, r_min=3, r_max=50):
    phi_step = (2 * np.pi) / phi_bins
    rspace = np.linspace(r_min, r_max, r_bins)
    mask = (xyz_pol[:, 0] >= r_min) & (xyz_pol[:, 0] <= r_max)
    rho_clipped = xyz_pol[mask, 0]
    phi_indices = ((xyz_pol[mask, 1] + np.pi) / phi_step).astype(int)
    phi_indices = np.clip(phi_indices, 0, phi_bins - 1)
    max_rho_per_bin = np.zeros(phi_bins, dtype=np.float32)
    np.maximum.at(max_rho_per_bin, phi_indices, rho_clipped)
    distance_feature = (rspace[None] <= max_rho_per_bin[:, None])
    return distance_feature.T


class PcMapLocDataset(data.Dataset):
    TARGET_KEYS = ['road_fork', 'building_corner', 'trunk', 'pole', 'traffic_sign']

    def __init__(self, opt, mode, use_osm_saliency: bool = True, use_pc_saliency: bool = True):
        self.opt = opt
        self.mode = mode
        self.tile_size = self.opt['tiling']['tile_margin']
        self.grid_size = np.asarray([480, 360, 32])

        with open('conf/semantic-kitti.yaml', 'r') as f:
            self.semantic_kitti = yaml.safe_load(f)
        self._learning_map = self.semantic_kitti['learning_map']
        max_label_key = max(self._learning_map.keys()) if self._learning_map else 0
        self._label_lut = np.zeros(max_label_key + 2, dtype=np.int64)
        for k, v in self._learning_map.items():
            self._label_lut[k] = v

        self.use_osm_saliency = use_osm_saliency
        self.use_pc_saliency = use_pc_saliency
        split_name = 'train' if mode == 'train' else 'test'
        self.osm_node_loader = NodeLoader(lambda s: _get_kitti_cfg(opt, s, split_name)['osm_node_pkl_path']) if use_osm_saliency else None
        self.pc_node_loader = NodeLoader(lambda s: _get_kitti_cfg(opt, s, split_name)['pc_node_pkl_path']) if use_pc_saliency else None

        self.data_list, self.tile_manager = self._make_dataset(mode)
        self._max_bound = np.array([50, np.pi, 1.5])
        self._min_bound = np.array([3, -np.pi, -3])
        self._intervals = (self._max_bound - self._min_bound) / (self.grid_size - 1)
        logger.info(f"[Dataset-{mode}] {len(self.data_list)} samples, osm_saliency={use_osm_saliency}, pc_saliency={use_pc_saliency}")

    def _make_dataset(self, mode='train'):
        if mode == 'train':
            sequence_list = self.opt['loading']['dataset_split']['train']
            split_name = 'train'
        elif mode in ('val', 'eval'):
            sequence_list = self.opt['loading']['dataset_split']['val']
            split_name = 'test'
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'train' or 'val'.")

        dataset = []
        tile_manager = {}
        for seq in sequence_list:
            seq = _zseq(seq)
            kcfg = _get_kitti_cfg(self.opt, seq, split_name)
            pose_npy_path = kcfg['pose_npy_path']
            if not os.path.exists(pose_npy_path):
                raise FileNotFoundError(f"[Dataset-{mode}] GPS file not found: {pose_npy_path}")
            pc_gps_file = np.load(pose_npy_path, allow_pickle=True).item()
            pc_bin_dir = kcfg['pc_bin_dir']
            label_dir = kcfg['pc_label_dir']
            if not os.path.isdir(pc_bin_dir):
                raise FileNotFoundError(f"[Dataset-{mode}] PC bin dir not found: {pc_bin_dir}")
            if not os.path.isdir(label_dir):
                raise FileNotFoundError(f"[Dataset-{mode}] label dir not found: {label_dir}")

            tile_path = kcfg['tile_pkl_path']
            if not os.path.exists(tile_path):
                self._save_seq_tile_manager(seq, pc_gps_file, split_name)
                if not os.path.exists(tile_path):
                    raise FileNotFoundError(f"[Dataset-{mode}] Tile manager not found after creation: {tile_path}")
            seq_tile_manager = self._load_seq_tile_manager(seq, split_name)

            for index, (lat, lon) in enumerate(zip(pc_gps_file['lat'], pc_gps_file['lon'])):
                pc_file_path = os.path.join(pc_bin_dir, f'{index:010d}.bin')
                label_path = os.path.join(label_dir, f'{index:010d}.label')
                if not os.path.exists(pc_file_path) or not os.path.exists(label_path):
                    continue
                dataset.append({
                    'seq': seq, 'index': index, 'pc_file_path': pc_file_path,
                    'label_path': label_path,
                    'gps_trans_data': pc_gps_file['T_w_velo_i'][index] if 'T_w_velo_i' in pc_gps_file else None,
                    'lat': float(lat), 'lon': float(lon)
                })
            tile_manager[seq] = seq_tile_manager
        return dataset, tile_manager

    def _save_seq_tile_manager(self, seq, pc_gps_file, split_name):
        seq_latlon = np.stack([pc_gps_file['lat'], pc_gps_file['lon']], 1)
        tile_margin = self.opt['tiling']['tile_margin']
        projection = Projection.from_points(seq_latlon)
        seq_xy = projection.project(seq_latlon)
        bbox_map_min = np.floor(seq_xy.min(0) / tile_margin) * tile_margin
        bbox_map_max = np.ceil(seq_xy.max(0) / tile_margin) * tile_margin
        seq_bbox = BoundaryBox(bbox_map_min, bbox_map_max) + tile_margin
        osm_path = _get_kitti_cfg(self.opt, seq, split_name)['osm_file_path']
        if not os.path.exists(osm_path):
            raise FileNotFoundError(f"Offline OSM file not found: {osm_path}")
        seq_tile_manager = TileManager.from_bbox(projection, seq_bbox, 2, path=Path(osm_path), tile_size=tile_margin)
        save_path = _get_kitti_cfg(self.opt, seq, split_name)['tile_pkl_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        seq_tile_manager.save(save_path)

    def _load_seq_tile_manager(self, seq, split_name):
        return TileManager.load(Path(_get_kitti_cfg(self.opt, seq, split_name)['tile_pkl_path']))

    @staticmethod
    def cart2polar(input_xyz):
        rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

    @staticmethod
    def random_rot90(raster, seed=None):
        if seed is None:
            rot = np.random.randint(0, 4)
        else:
            rot = np.random.RandomState(seed).randint(0, 4)
        raster = np.rot90(raster, rot, axes=(-2, -1))
        return raster, rot

    @staticmethod
    def random_flip(raster, seed=None):
        state = np.random if seed is None else np.random.RandomState(seed)
        flip_x = state.rand() > 0.5
        flip_y = False
        if not flip_x:
            flip_y = state.rand() > 0.5
        if flip_x:
            raster = raster[..., :, ::-1]
        elif flip_y:
            raster = raster[..., ::-1, :]
        return raster, (flip_x, flip_y)

    @staticmethod
    def augment_point_cloud_with_2d_rotation(point_cloud, theta=None):
        x, y = point_cloud[:, 0], point_cloud[:, 1]
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_xy = np.dot(np.column_stack((x, y)), rotation_matrix.T)
        point_cloud[:, 0] = rotated_xy[:, 0]
        point_cloud[:, 1] = rotated_xy[:, 1]
        return point_cloud, theta

    def _create_semantic_bev(self, grid_ind, labels, R_bins=480, Phi_bins=360):
        r_idx = np.clip(grid_ind[:, 0], 0, R_bins - 1).astype(np.int32)
        phi_idx = np.clip(grid_ind[:, 1], 0, Phi_bins - 1).astype(np.int32)
        labels_flat = np.clip(labels.flatten().astype(np.int32), 0, 19)
        return create_semantic_bev_numba(r_idx, phi_idx, labels_flat, R_bins, Phi_bins)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = self.data_list[index]
        seq, seq_index, pc_file_path, label_path = data_item['seq'], data_item['index'], data_item['pc_file_path'], data_item['label_path']
        lat, lon = data_item['lat'], data_item['lon']

        seq_tile_manager = self.tile_manager[seq]
        latlon = np.array([lat, lon])
        xy = seq_tile_manager.projection.project(latlon)
        sample_bbox = BoundaryBox(xy - self.tile_size // 2, xy + self.tile_size // 2)
        canvas = seq_tile_manager.query(sample_bbox)
        raster = canvas.raster
        pcs = np.fromfile(pc_file_path, dtype=np.float32).reshape(-1, 4)
        aug_params = {'rot_k': 0, 'flip_x': False, 'flip_y': False, 'theta': 0.0}
        if self.mode == 'train':
            raster, rot_k = self.random_rot90(raster)
            aug_params['rot_k'] = rot_k
            raster, (flip_x, flip_y) = self.random_flip(raster)
            aug_params['flip_x'] = flip_x
            aug_params['flip_y'] = flip_y
            pcs, theta = self.augment_point_cloud_with_2d_rotation(pcs)
            aug_params['theta'] = theta

        raw_label = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        inline_mask = (np.sqrt(pcs[:, 0] ** 2 + pcs[:, 1] ** 2) < 50) & (np.abs(pcs[:, 2]) < 3)
        intensity = pcs[:, 3][inline_mask]
        pcs = pcs[inline_mask]
        raw_label = raw_label[inline_mask]
        raw_label_clipped = np.clip(raw_label, 0, len(self._label_lut) - 1)
        labels = self._label_lut[raw_label_clipped].reshape((-1, 1))
        xyz_pol = self.cart2polar(pcs)
        grid_ind = (np.floor((np.clip(xyz_pol, self._min_bound, self._max_bound) - self._min_bound) / self._intervals)).astype(np.int32)
        distance_feature_2d = compute_distance_feature_polar(xyz_pol)
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * self._intervals + self._min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, pcs[:, :2]), axis=1)
        semantic_bev = self._create_semantic_bev(grid_ind, labels)
        data_tuple = (grid_ind, labels, return_xyz, intensity)

        if self.use_osm_saliency and self.osm_node_loader is not None:
            osm_nodes_raw = self.osm_node_loader.get_nodes_raw(seq, seq_index)
            osm_node_coords_dict = {}
            for key in self.TARGET_KEYS:
                c = osm_nodes_raw.get(key, [])
                if len(c) > 0:
                    c = np.array(c, dtype=np.float32)
                    if c.ndim == 1:
                        c = c.reshape(1, -1)
                    osm_node_coords_dict[key] = c[:, :2] - xy[None, :]
                else:
                    osm_node_coords_dict[key] = np.zeros((0, 2), dtype=np.float32)
        else:
            osm_node_coords_dict = {k: np.zeros((0, 2), dtype=np.float32) for k in self.TARGET_KEYS}

        if self.use_pc_saliency and self.pc_node_loader is not None:
            pc_nodes_raw = self.pc_node_loader.get_nodes_raw(seq, seq_index)
            pc_node_coords_dict = {}
            for key in self.TARGET_KEYS:
                c = pc_nodes_raw.get(key, [])
                if len(c) > 0:
                    c = np.array(c, dtype=np.float32)
                    if c.ndim == 1:
                        c = c.reshape(1, -1)
                    pc_node_coords_dict[key] = c[:, :2]
                else:
                    pc_node_coords_dict[key] = np.zeros((0, 2), dtype=np.float32)
        else:
            pc_node_coords_dict = {k: np.zeros((0, 2), dtype=np.float32) for k in self.TARGET_KEYS}

        return {
            'data_tuple': data_tuple,
            'dataset_index': index,
            'osm_map': torch.from_numpy(np.ascontiguousarray(raster)).long(),
            'xy': torch.from_numpy(xy.astype(np.float32)),
            'pc_vis_mask': torch.from_numpy(distance_feature_2d.astype(np.float32)) / 50,
            'semantic_bev': torch.from_numpy(semantic_bev).long(),
            'osm_node_coords': osm_node_coords_dict,
            'pc_node_coords': pc_node_coords_dict,
            'aug_params': aug_params,
        }


def collate_fn_BEV(data):
    grid_ind_stack = [d['data_tuple'][0] for d in data]
    point_label = [d['data_tuple'][1] for d in data]
    xyz = [d['data_tuple'][2] for d in data]
    intensity = [d['data_tuple'][3] for d in data]
    osm_map = torch.stack([d['osm_map'] for d in data])
    xy = torch.stack([d['xy'] for d in data])
    pc_vis_mask = torch.stack([d['pc_vis_mask'] for d in data])
    semantic_bev = torch.stack([d['semantic_bev'] for d in data])
    osm_node_coords_list = [{k: torch.from_numpy(v).float() for k, v in d['osm_node_coords'].items()} for d in data]
    pc_node_coords_list = [{k: torch.from_numpy(v).float() for k, v in d['pc_node_coords'].items()} for d in data]
    aug_params_list = [d['aug_params'] for d in data]
    dataset_indices = [d['dataset_index'] for d in data]
    return (grid_ind_stack, point_label, xyz, osm_map, xy, pc_vis_mask,
            intensity, semantic_bev, osm_node_coords_list, pc_node_coords_list,
            aug_params_list, dataset_indices)