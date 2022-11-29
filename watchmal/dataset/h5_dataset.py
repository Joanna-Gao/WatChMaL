"""
Class for loading data in h5 format
"""

# torch imports
from torch.utils.data import Dataset

# generic imports
import h5py
import numpy as np
from abc import ABC, abstractmethod

class H5CommonDataset(Dataset, ABC):
    """
    Initialize with file of h5 data.  Sets up access to all of the data that is
    common between the digitized hits data and the truth hits data.  These are:
    
    event_ids 	(n_events,) 	int32 	ID of the event in the ROOT file
    root_files 	(n_events,) 	object 	File name and location of the ROOT file
    labels 	    (n_events,) 	int32 	Label for event classification 
                                      (gamma=0, electron=1, muon=2)
    positions 	(n_events,1,3) 	float32 	Initial (x, y, z) position of 
                                          simulated particle
    angles 	    (n_events,2) 	float32 Initial direction of simulated particle as
                                      (polar, azimuth) angles
    energies 	  (n_events,1) 	float32	Initial total energy of simulated particle
    veto 	      (n_events,) 	bool 	  OD veto estimate based on any Cherenkov 
                                      producing particles exiting the tank, with
                                      initial energy above threshold
    veto2     	(n_events,) 	bool 	  OD veto estimate based on any Cherenkov 
                                      producing particles exiting the tank, with
                                      an estimate of energy at the point the
                                      particle exits the tank being above 
                                      threshold
    event_hits_index 	(n_events,)   int64  Starting index in the hit dataset 
                                           objects for hits of a particular event
    
    hit_pmt 	  (n_hits,) 	  int32 	  PMT ID of the digitized hit
    hit_time 	  (n_hits,) 	  float32 	Time of the digitized hit

    Note: event_hits_index, hit_pmt and hit_time have to be separate for 20" 
          and 3" PMTs
    """
    def __init__(self, h5_path, is_distributed):
        """
        Args:
            h5_path             ... path to h5 dataset file
            is_distributed      ... whether running in multiprocessing mode
            transforms          ... transforms to apply
        """
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_length = h5_file["labels"].shape[0]

        self.initialized = False
        if not is_distributed:
            self.initialize()

    def initialize(self):
        self.file_descriptor = open(self.h5_path, 'rb')
        self.h5_file = h5py.File(self.file_descriptor, "r")

        self.event_ids  = np.array(self.h5_file["event_ids"])
        self.root_files = np.array(self.h5_file["root_files"])
        self.labels     = np.array(self.h5_file["labels"])
        self.positions  = np.array(self.h5_file["positions"])
        self.angles     = np.array(self.h5_file["angles"])
        self.energies   = np.array(self.h5_file["energies"])
        if "veto" in self.h5_file.keys():
            self.veto  = np.array(self.h5_file["veto"])
            self.veto2 = np.array(self.h5_file["veto2"])
        self.event_hits_index_20 = np.append(self.h5_file["event_hits_index_20"],\
            self.h5_file["hit_pmt_20"].shape[0]).astype(np.int64)  # why append this? FIXME!
        self.event_hits_index_3 = np.append(self.h5_file["event_hits_index_3"],\
            self.h5_file["hit_pmt_3"].shape[0]).astype(np.int64)

        self.hdf5_hit_pmt_20  = self.h5_file["hit_pmt_20"]
        self.hdf5_hit_time_20 = self.h5_file["hit_time_20"]

        self.hdf5_hit_pmt_3  = self.h5_file["hit_pmt_3"]
        self.hdf5_hit_time_3 = self.h5_file["hit_time_3"]

        self.hit_pmt_20 = np.memmap(self.h5_path, mode="r", 
                                    shape=self.hdf5_hit_pmt_20.shape,
                                    offset=self.hdf5_hit_pmt_20.id.get_offset(),
                                    dtype=self.hdf5_hit_pmt_20.dtype)
        self.hit_pmt_3 = np.memmap(self.h5_path, mode="r", 
                                   shape=self.hdf5_hit_pmt_3.shape,
                                   offset=self.hdf5_hit_pmt_3.id.get_offset(),
                                   dtype=self.hdf5_hit_pmt_3.dtype)

        self.time_20 = np.memmap(self.h5_path, mode="r", 
                                 shape=self.hdf5_hit_time_20.shape,
                                 offset=self.hdf5_hit_time_20.id.get_offset(),
                                 dtype=self.hdf5_hit_time_20.dtype)
        self.time_3 = np.memmap(self.h5_path, mode="r", 
                                shape=self.hdf5_hit_time_3.shape,
                                offset=self.hdf5_hit_time_3.id.get_offset(),
                                dtype=self.hdf5_hit_time_3.dtype)
        self.load_hits()

        # Set attribute so that method won't be invoked again
        self.initialized = True
        
    @abstractmethod
    def load_hits(self):  
        # place holder, this method awaits to be redefined by the subclasses
        pass

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        if not self.initialized:
            self.initialize()

        data_dict = {
            "labels": self.labels[item],
            "energies": self.energies[item],
            "angles": self.angles[item],
            "positions": self.positions[item],
            "event_ids": self.event_ids[item],
            "root_files": self.root_files[item],
            "indices": item
        }

        return data_dict


class H5Dataset(H5CommonDataset, ABC):
    """
    Initialize digihits dataset.  Adds access to digitized hits data. These are:
    hit_charge 	(n_hits,) 	float32 	Charge of the digitized hit
    
    Note: hit_charge has to be separate for 20" and 3" PMTs
    """
    def __init__(self, h5_path, is_distributed):
        H5CommonDataset.__init__(self, h5_path, is_distributed)
        
    def load_hits(self):
        self.hdf5_hit_charge_20 = self.h5_file["hit_charge_20"]
        self.hit_charge_20 = np.memmap(self.h5_path, mode="r",
                                       shape=self.hdf5_hit_charge_20.shape,
                                       offset=self.hdf5_hit_charge_20.id.get_offset(),
                                       dtype=self.hdf5_hit_charge_20.dtype)
        self.hdf5_hit_charge_3 = self.h5_file["hit_charge_3"]
        self.hit_charge_3 = np.memmap(self.h5_path, mode="r",
                                      shape=self.hdf5_hit_charge_3.shape,
                                      offset=self.hdf5_hit_charge_3.id.get_offset(),
                                      dtype=self.hdf5_hit_charge_3.dtype)
        
    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        start_20 = self.event_hits_index_20[item]
        stop_20 = self.event_hits_index_20[item + 1]

        self.event_hit_pmts_20 = self.hit_pmt_20[start_20:stop_20].astype(np.int16)
        self.event_hit_charges_20 = self.hit_charge_20[start_20:stop_20]
        self.event_hit_times_20 = self.time_20[start_20:stop_20]

        start_3 = self.event_hits_index_3[item]
        stop_3 = self.event_hits_index_3[item + 1]

        self.event_hit_pmts_3 = self.hit_pmt_3[start_3:stop_3].astype(np.int16)
        self.event_hit_charges_3 = self.hit_charge_3[start_3:stop_3]
        self.event_hit_times_3 = self.time_3[start_3:stop_3]

        return data_dict

class H5TrueDataset(H5CommonDataset, ABC):
    """
    Initializes truehits dataset. Adds access to true photon hits data. 
    These are:
    hit_parent 	  (n_hits,) 	float32 	Parent track ID of the true hit, as 
                                        defined by WCSim's true hit parent. 
                                        -1 is used for dark noise.

    Note: modified for hybrid HK geometry.
    """
    def __init__(self, h5_path, transforms=None, digitize_hits=True):
        H5CommonDataset.__init__(self, h5_path, transforms)
        self.digitize_hits = digitize_hits

    def load_hits(self):
        self.all_hit_parent_20 = self.h5_file["hit_parent_20"]
        self.hit_parent_20 = np.memmap(self.h5_path, mode="r",
                                       shape=self.all_hit_parent_20.shape,
                                       offset=self.all_hit_parent_20.id.get_offset(),
                                       dtype=self.all_hit_parent_20.dtype)

        self.all_hit_parent_3 = self.h5_file["hit_parent_3"]
        self.hit_parent_3 = np.memmap(self.h5_path, mode="r",
                                       shape=self.all_hit_parent_3.shape,
                                       offset=self.all_hit_parent_3.id.get_offset(),
                                       dtype=self.all_hit_parent_3.dtype)

    def digitize(self, truepmts, truetimes, trueparents):
        """
        Replace below with a real digitization.  For now take time closest to zero as time, and sum of photons as charge.
        """
        pmt_time_dict = { pmt: truetimes[ truepmts==pmt ] for pmt in truepmts }
        pmt_photons_dict = { pmt : len(truetimes[ truepmts==pmt]) for pmt in truepmts }
        pmt_mintimes_dict = { pmt : min( abs(truetimes[ truepmts==pmt]) )   for pmt in truepmts }

        timeoffset = 950.0
        allpmts  = np.array( list(pmt_photons_dict.keys()) )
        alltimes = np.array( list(pmt_mintimes_dict.values()) ) + timeoffset
        allcharges = np.array( list(pmt_photons_dict.values()) )
        return allpmts, alltimes, allcharges

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        start_20 = self.event_hits_index_20[item]
        stop_20 = self.event_hits_index_20[item + 1]

        true_pmts_20    = self.hit_pmt_20[start_20:stop_20].astype(np.int16)
        true_times_20   = self.time_20[start_20:stop_20]
        true_parents_20 = self.hit_parent_20[start_20:stop_20]

        start_3 = self.event_hits_index_3[item]
        stop_3 = self.event_hits_index_3[item + 1]
        true_pmts_3    = self.hit_pmt_3[start_3:stop_3].astype(np.int16)
        true_times_3   = self.time_3[start_3:stop_3]
        true_parents_3 = self.hit_parent_3[start_3:stop_3]

        if self.digitize_hits:
            self.event_hit_pmts_20, self.event_hit_times_20, self.event_hit_charges_20 = self.digitize(true_pmts_20, true_times_20, true_parents_20)
            self.event_hit_pmts_3, self.event_hit_times_3, self.event_hit_charges_3 = self.digitize(true_pmts_3, true_times_3, true_parents_3)
        else:
            self.event_hit_pmts_20 = true_pmts_20
            self.event_hit_times_20 = true_times_20
            self.event_hit_parents_20 = true_parents_20

            self.event_hit_pmts_3 = true_pmts_3
            self.event_hit_times_3 = true_times_3
            self.event_hit_parents_3 = true_parents_3

        return data_dict