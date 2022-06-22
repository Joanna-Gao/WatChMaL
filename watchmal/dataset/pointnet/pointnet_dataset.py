"""
Class implementing a mPMT dataset for pointnet in h5 format
"""

# torch imports
import torch

# generic imports
import numpy as np

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du


class PointNetDataset(H5Dataset):

    def __init__(self, h5file, geometry_file, is_distributed, 
                 use_orientations=False, n_points_20=8000, n_points_3=8000, 
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
            data = np.zeros((5, self.n_points_20+self.n_points_3))
        else:
            # For some reason the orientation is a (n, 3) matrix when it was
            # first extracted from the root file, but here it became a (n, 2)
            # matrix. I vaguly remembered it's converted to zenith and azmith
            # angles at some point but I can't find the code anywhere. So
            # probably best not to use this function.....
            hit_orientations_20 = \
                self.geo_orientations_20[self.event_hit_pmts_20[:n_hits_20], :]
            hit_orientations_3 = \
                self.geo_orientations_3[self.event_hit_pmts_3[:n_hits_3], :]
            data = np.zeros((8, self.n_points_20+self.n_points_3))
            data[3:5, :n_hits_20] = hit_orientations_20.T
            data[3:5, self.n_points_20:(self.n_points_20+n_hits_3)] = \
                hit_orientations_3.T 

        # The data is store in the following way:
        #        ----------20in----------|----------3in----------
        #   x    * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0 
        #   y    * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0
        #   z    * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0
        # (ori 1 * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0)
        # (ori 2 * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0)  
        # charge * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0 
        #  time  * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0
        # label  0 0 0 0 0 ... 0 0 0 0 0 | 1 1 1 1 ... 1 1 1 1 1
        #        ---hits---|---no hits---|---hits---|---no hits--
        #        0         ...      7999 | 8000    ...      15999
        # where the *'s are the numbers, the bracketed rows are omitable
        # depending on if you want to use the wcsim outputed PMT orientation or
        # not (not recommended). Max number of 20in hits is n_points_20, 
        # currently set at 8000, and max number of 3in hits is n_points_3 = 8000

        # 20"
        data[:3, :n_hits_20] = hit_positions_20[:n_hits_20].T
        data[-3, :n_hits_20] = self.event_hit_charges_20[:n_hits_20]
        data[-2, :n_hits_20] = self.event_hit_times_20[:n_hits_20]
        data[-1, :n_hits_20] = [0] * n_hits_20
        # 3"
        index_3 = self.n_points_20 + n_hits_3
        data[:3, self.n_points_20:index_3] = hit_positions_3[:n_hits_3].T
        data[-3, self.n_points_20:index_3] = self.event_hit_charges_3[:n_hits_3]
        data[-2, self.n_points_20:index_3] = self.event_hit_times_3[:n_hits_3]
        data[-1, self.n_points_20:index_3] = [1] * n_hits_3

        data = du.apply_random_transformations(self.transforms, data)

        data_dict["data"] = data
        return data_dict