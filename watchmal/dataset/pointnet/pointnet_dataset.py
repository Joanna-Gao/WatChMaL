"""
Class implementing a mPMT dataset for pointnet in h5 format
"""

# generic imports
import numpy as np

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du


class PointNetDataset(H5Dataset):

    def __init__(self, h5file, geometry_file, is_distributed, 
                 use_orientations=False, n_points_20=4000, n_points_3=4000, 
                 transforms=None):
        """
        Using the separate geo info format for the hybrid geometry
        The 'n_points' value might need to change FIXME!! 
        """
        super().__init__(h5file, is_distributed)
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions_20 = torch.from_numpy(geo_file["position_20"]).float()
        self.geo_orientations_20 = torch.from_numpy(geo_file["orientation_20"]).float()
        self.geo_positions_3 = torch.from_numpy(geo_file["position_3"]).float()
        self.geo_orientations_3 = torch.from_numpy(geo_file["orientation_3"]).float()
        self.use_orientations = use_orientations
        self.n_points_20 = n_points_20
        self.n_points_3 = n_points_3
        self.transforms = du.get_transformations(transformations, transforms)

    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)

        hit_positions_20 = self.geo_positions_20[self.event_hit_pmts_20, :]
        n_hits_20 = min(self.n_points_20, self.event_hit_pmts_20.shape[0])
        hit_positions_3 = self.geo_positions_3[self.event_hit_pmts_3, :]
        n_hits_3 = min(self.n_points_3, self.event_hit_pmts_3.shape[0])
        if not self.use_orientations:
            data = np.zeros((5, self.n_points_20+self.n_points_3))  # number correct?? FIXME
        else:
            hit_orientations_20 = self.geo_orientations_20[self.event_hit_pmts_20[:n_hits_20], :]
            hit_orientations_3 = self.geo_orientations_3[self.event_hit_pmts_3[:n_hits_3], :]
            data = np.zeros((7, self.n_points_20+self.n_points_3))  # number correct?? FIXME
            data[3:5, :n_hits_20] = hit_orientations_20.T
            data[3:5, self.n_points_20:n_hits_3] = hit_orientations_3.T  # 3"
        # 20"
        data[:3, :n_hits_20] = hit_positions_20[:n_hits_20].T
        data[-2, :n_hits_20] = self.event_hit_charges_20[:n_hits_20]
        data[-1, :n_hits_20] = self.event_hit_times_20[:n_hits_20]
        # 3"
        data[:3, self.n_points_20:n_hits_3] = hit_positions_3[:n_hits_3].T
        data[-2, self.n_points_20:n_hits_3] = self.event_hit_charges_3[:n_hits_3]
        data[-1, self.n_points_20:n_hits_3] = self.event_hit_times_3[:n_hits_3]

        data = du.apply_random_transformations(self.transforms, data)

        data_dict["data"] = data
        return data_dict