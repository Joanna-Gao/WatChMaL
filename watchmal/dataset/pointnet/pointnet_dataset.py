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

    def __init__(self, h5file, geometry_file, is_distributed, use_times=True, 
                 use_orientations=False, n_points_20=4000, n_points_3=4000,
                 transforms=None):
        super().__init__(h5file, is_distributed)
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions_20 = geo_file["position_20"].astype(np.float32)
        self.geo_orientations_20 = geo_file["orientation_20"].astype(np.float32)
        self.geo_positions_3 = geo_file["position_3"].astype(np.float32)
        self.geo_orientations_3 = geo_file["orientation_3"].astype(np.float32)
        self.use_orientations = use_orientations
        self.use_times = use_times
        self.n_points_20 = n_points_20
        self.n_points_3 = n_points_3
        self.transforms = du.get_transformations(transformations, transforms)
        self.channels = 5  #x,y,z,q,pmt flag
        if use_orientations:
            self.channels += 3
        if use_times:
            self.channels += 1

    def  __getitem__(self, item):

        data_dict = super().__getitem__(item)

        n_hits_20 = min(self.n_points_20, self.event_hit_pmts_20.shape[0])
        hit_positions_20 = self.geo_positions_20[self.event_hit_pmts_20[:n_hits_20], :]
        n_hits_3 = min(self.n_points_3, self.event_hit_pmts_3.shape[0])
        hit_positions_3 = self.geo_positions_3[self.event_hit_pmts_3[:n_hits_3], :]
        data = np.zeros((self.channels, self.n_points_20+self.n_points_3), dtype=np.float32)

        # The data is store in the following way:
        #        ----------20in----------|----------3in----------
        #   x    * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0 
        #   y    * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0
        #   z    * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0
        # (ori 1 * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0)
        # (ori 2 * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0)  
        #  time  * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0 
        # charge * * * * * ... 0 0 0 0 0 | * * * * ... 0 0 0 0 0
        # label  1 1 1 1 1 ... 0 0 0 0 0 | 2 2 2 2 ... 0 0 0 0 0
        #        ---hits---|---no hits---|---hits---|---no hits--
        #        0         ...      3999 | 4000    ...      7999
        # where the *'s are the numbers, the bracketed rows are omitable
        # depending on if you want to use the wcsim outputed PMT orientation or
        # not (not recommended). Max number of 20in hits is n_points_20, 
        # currently set at 4000, and max number of 3in hits is n_points_3 = 4000

        # 20"
        data[:3, :n_hits_20] = hit_positions_20[:n_hits_20].T
        data[-1, :n_hits_20] = [1] * n_hits_20

        # 3"
        index_3 = self.n_points_20 + n_hits_3
        data[:3, self.n_points_20:index_3] = hit_positions_3[:n_hits_3].T
        data[-1, self.n_points_20:index_3] = [2] * n_hits_3

        if self.use_orientations:
            hit_orientations_20 = self.geo_orientations_20[self.event_hit_pmts_20[:n_hits_20], :]
            data[3:6, :n_hits_20] = hit_orientations_20.T
            hit_orientations_3 = self.geo_orientations_3[self.event_hit_pmts_3[:n_hits_3], :]
            data[3:6, n_hits_20:index_3] = hit_orientations_3.T
        if self.use_times:
            data[-3, :n_hits_20] = self.event_hit_times_20[:n_hits_20]
            data[-3, self.n_points_20:index_3] = self.event_hit_times_3[:n_hits_3]

        data[-2, :n_hits_20] = self.event_hit_charges_20[:n_hits_20]
        data[-2, self.n_points_20:index_3] = self.event_hit_charges_3[:n_hits_3]

        data = du.apply_random_transformations(self.transforms, data)

        data_dict["data"] = data
        return data_dict