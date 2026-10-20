import os
import torch.utils.data
import h5py
from contextlib import contextmanager
import numpy as np
import json
import random
from collections import OrderedDict
import torch.utils.data
import torch
from typing import Dict, Callable

from termcolor import cprint

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.utils.dataset import action_stats_to_normalization_stats, _compute_traj_stats, _aggregate_traj_stats

def convert_to_weight(dist, sharp=50, cutoff=0.001):
    weights = 1 / (1 + np.exp(sharp * (dist - cutoff)))
    return weights

def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


class ChunkSamplingDTWTrainDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_to_paths,
            pair_info_path,
            sharpness,
            cutoff,

            window_size,
            dataset_masks,

            obs_keys,
            action_keys,
            dataset_keys,
            action_config,

            frame_stack=1,
            seq_length=1,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=True,
            shuffled_obs_key_groups=None,
            lang=None,
            demo_limit=None,
            no_window=False,
    ):
        """
		Dataset class for fetching sequences of experience.
		Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

		Args:
			hdf5_path (str): path to hdf5

			obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset

			dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

			frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).

			seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

			pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
				ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
				first frame stacked observation would be (s_0, s_1, s_2, s_3).

			pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
				ensures that partial sequences at the end of a demonstration are observed, such as
				(s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
				(s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

			get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
				useful for masking loss functions on padded parts of the data.

			goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals

			hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
				that multiple Dataset instances can all access the same hdf5 file without problems.

			hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
				and std of each observation (in each dimension and modality), and normalizing to unit
				mean and variance in each dimension.

			filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
				demonstrations to load

			load_next_obs (bool): whether to load next_obs from the dataset
		"""

        super(ChunkSamplingDTWTrainDataset, self).__init__()

        self.dataset_paths = dataset_to_paths # dict
        self.dataset_masks = dataset_masks

        self.sharpness = sharpness
        self.cutoff = cutoff

        self.window_size = window_size

        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_files = {}

        self.load_next_obs = load_next_obs
        self.pair_info_path = pair_info_path

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.action_keys = tuple(action_keys)
        self.dataset_keys = tuple(dataset_keys)
        # add action keys to dataset keys
        if self.action_keys is not None:
            self.dataset_keys = tuple(set(self.dataset_keys).union(set(self.action_keys)))

        self.action_config = action_config

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.num_src_traj = None

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"  # we use last next_obs as goal

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info()

        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        self.action_normalization_stats = None
        self._action_stats_cache = None

        if shuffled_obs_key_groups is None:
            self.shuffled_obs_key_groups = list()
        else:
            self.shuffled_obs_key_groups = shuffled_obs_key_groups

        self.no_window = no_window

        # maybe prepare for observation normalization
        # maybe store dataset in memory for fast access

        self.close_and_delete_hdf5_handle()

    def load_demo_info(self,):
        """
		Args:
			filter_by_attribute (str): if provided, use the provided filter key
				to select a subset of demonstration trajectories to load

			demos (list): list of demonstration keys to load from the hdf5 file. If
				omitted, all demos in the file (or under the @filter_by_attribute
				filter key) are used.
		"""

        # filter demo trajectory by mask
        assert len(self.hdf5_files.items()) == 2


        self.demos = []
        self.src_demos = None

        for k, v in self.hdf5_files.items():

            this_demos = list(v["data"].keys())
            random.shuffle(this_demos)
            if (self.dataset_masks is not None):
                if k in self.dataset_masks:
                    this_demos = this_demos[:self.dataset_masks[k]]

            inds = np.argsort([int(elem[5:]) for elem in this_demos])
            cprint(f"Dataset: {k} - Demos: {len(inds)}", "cyan")

            if k == "ot_src":
                assert self.src_demos is None
                self.src_demos = [this_demos[i] for i in inds]
            elif k == "ot_tgt":
                self.demos += [[this_demos[i], k] for i in inds]
            else:
                raise

        pair_info = json.load(open(os.path.expanduser(self.pair_info_path), "r"))
        tgt_demo_id_to_demo_and_weight = {}
        paired_idx = {}
        for tgt_demo_id, pairs in pair_info.items():
            demo_names = []
            dists = []
            for one_pair in pairs:
                src_demo_id = one_pair["demo_name"]
                if src_demo_id not in self.src_demos:
                    continue

                demo_names.append(src_demo_id)
                dists.append(one_pair["raw_dtw_dist"])
                idx_map = {int(k): v for k, v in one_pair["pairing"].items()}
                paired_idx[f"{tgt_demo_id}-{src_demo_id}"] = idx_map

            dists = np.array(dists)
            weights = convert_to_weight(dists, sharp=self.sharpness, cutoff=self.cutoff)
            weights /= weights.sum()

            tgt_demo_id_to_demo_and_weight[tgt_demo_id] = [demo_names, weights]

        self.tgt_name_to_demo_and_weight = tgt_demo_id_to_demo_and_weight
        self.paired_idx = paired_idx

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._index_to_dataset_name = dict()

        self._dataset_demo_id_to_start_indices = dict()  # gives start index per demo id
        self._dataset_demo_id_to_demo_length = dict()

        self._src_dataset_demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0

        for ep, dataset_name in self.demos:
            demo_length = self.hdf5_files[dataset_name]["data/{}".format(ep)].attrs["num_samples"]
            demo_key = f"{dataset_name}:{ep}"
            self._dataset_demo_id_to_start_indices[demo_key] = self.total_num_sequences
            self._dataset_demo_id_to_demo_length[demo_key] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self._index_to_dataset_name[self.total_num_sequences] = dataset_name
                self.total_num_sequences += 1

        # for ep, dataset_name in self.src_demos:
        #     demo_length = self.hdf5_files[dataset_name]["data/{}".format(ep)].attrs["num_samples"]
        #     demo_key = f"{dataset_name}:{ep}"
        #     self._src_dataset_demo_id_to_demo_length[demo_key] = demo_length

        cprint(f"Selected demos: {self.n_demos}\nTotal Samples: {self.total_num_sequences}", "red")

    @property
    def hdf5_files(self):
        """
		This property allows for a lazy hdf5 file open.
		"""
        if len(self._hdf5_files) == 0:
            for dataset_name in self.dataset_paths:
                dataset_path = os.path.expanduser(self.dataset_paths[dataset_name])
                self._hdf5_files[dataset_name] = h5py.File(dataset_path, 'r',
                                                           swmr=self.hdf5_use_swmr, libver='latest')

        return self._hdf5_files

    def close_and_delete_hdf5_handle(self):
        """
		Maybe close the file handle.
		"""
        for k, v in self._hdf5_files.items():
            v.close()
        self._hdf5_files = {}

    @contextmanager
    def hdf5_file_opened(self):
        """
		Convenient context manager to open the file on entering the scope
		and then close it on leaving.
		"""
        should_close = len(self.hdf5_files) == 0
        yield self.hdf5_files
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
		Pretty print the class and important attributes on a call to `print`.
		"""
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = getattr(self, "filter_by_attribute", None) or "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        cache_mode_str = "none"
        msg = msg.format(self.dataset_paths, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        """
		Ensure that the torch dataloader will do a complete pass through all sequences in
		the dataset before starting a new iteration.
		"""
        return self.total_num_sequences

    def get_dataset_for_ep(self, ep, dataset_name, key):
        """
		Helper utility to get a dataset for a specific demonstration.
		Takes into account whether the dataset has been loaded into memory.
		"""

        # check if this key should be in memory
        hd5key = "data/{}/{}".format(ep, key)
        ret = self.hdf5_files[dataset_name][hd5key]
        return ret

    def __getitem__(self, index):
        """
		Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
		"""
        data = self.get_item(index)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_item_from_dataset(self, dataset_name, demo_id, index_in_demo):
        """
        Main implementation of getitem when not using cache.
        """

        demo_length = self.hdf5_files[dataset_name]["data/{}".format(demo_id)].attrs["num_samples"]

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            dataset_name=dataset_name,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1,  # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length

        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            dataset_name=dataset_name,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                dataset_name=dataset_name,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                dataset_name=dataset_name,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        # get action components
        ac_dict = OrderedDict()
        for k in self.action_keys:
            ac = meta[k]
            # expand action shape if needed
            if len(ac.shape) == 1:
                ac = ac.reshape(-1, 1)
            ac_dict[k] = ac

        # concatenate all action components (no normalization for OT dataset)
        meta["actions"] = AcUtils.action_dict_to_vector(ac_dict)

        # also return the sampled index
        # meta["index"] = f"{dataset_name}_{demo_id}_{index_in_demo}"

        # # language embedding
        # if self._lang_emb is not None:
        #     T = meta["actions"].shape[0]
        #     meta["obs"][LangUtils.LANG_EMB_OBS_KEY] = np.tile(self._lang_emb, (T, 1))

        return meta

    def get_item(self, index):
        """
		Main implementation of getitem when not using cache.
		"""

        tgt_demo_id = self._index_to_demo_id[index]
        tgt_demo_key = f"ot_tgt:{tgt_demo_id}"

        demo_start_index = self._dataset_demo_id_to_start_indices[tgt_demo_key]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        tgt_index_in_demo = index - demo_start_index + demo_index_offset

        tgt_meta = self.get_item_from_dataset(dataset_name="ot_tgt", demo_id=tgt_demo_id,
                                              index_in_demo=tgt_index_in_demo)

        src_demo_names, sample_weights = self.tgt_name_to_demo_and_weight[tgt_demo_id]
        src_demo_id = np.random.choice(src_demo_names, size=1, p=sample_weights)[0]
        src_demo_len = self.hdf5_files["ot_src"]["data/{}".format(src_demo_id)].attrs["num_samples"]

        idx_map = self.paired_idx[f"{tgt_demo_id}-{src_demo_id}"]
        idx_list = idx_map[tgt_index_in_demo]

        src_index_in_demo = np.random.choice(idx_list, size=1)[0]

        if not self.no_window:
            samp_src_index_in_demo = random.choice(range(src_index_in_demo - self.window_size,
                                                        src_index_in_demo + self.window_size))
            samp_src_index_in_demo = np.clip(samp_src_index_in_demo, a_min=0, a_max=src_demo_len - 1)
        else:
            samp_src_index_in_demo = random.choice(range(0, src_demo_len))

        src_meta = self.get_item_from_dataset(dataset_name="ot_src", demo_id=src_demo_id,
                                              index_in_demo=samp_src_index_in_demo)

        data = {
            "src": src_meta,
            "tgt": tgt_meta
        }

        return data

    def get_sequence_from_demo(self, demo_id, dataset_name, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
		Extract a (sub)sequence of data items from a demo given the @keys of the items.

		Args:
			demo_id (str): id of the demo, e.g., demo_0
			index_in_demo (int): beginning index of the sequence wrt the demo
			keys (tuple): list of keys to extract
			num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
			seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

		Returns:
			a dictionary of extracted items.
		"""
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        # demo_key = f"{dataset_name}:{demo_id}"
        # demo_length = self._dataset_demo_id_to_demo_length[demo_key]
        demo_length = self.hdf5_files[dataset_name]["data/{}".format(demo_id)].attrs["num_samples"]


        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(ep=demo_id, dataset_name=dataset_name, key=k)
            seq[k] = data[seq_begin_index: seq_end_index]

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, dataset_name, index_in_demo, keys, num_frames_to_stack=0, seq_length=1,
                                   prefix="obs"):
        """
		Extract a (sub)sequence of observation items from a demo given the @keys of the items.

		Args:
			demo_id (str): id of the demo, e.g., demo_0
			index_in_demo (int): beginning index of the sequence wrt the demo
			keys (tuple): list of keys to extract
			num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
			seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
			prefix (str): one of "obs", "next_obs"

		Returns:
			a dictionary of extracted items.
		"""
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id=demo_id,
            dataset_name=dataset_name,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        return obs

    def get_dataset_sequence_from_demo(self, demo_id, dataset_name, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
		Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).

		Args:
			demo_id (str): id of the demo, e.g., demo_0
			index_in_demo (int): beginning index of the sequence wrt the demo
			keys (tuple): list of keys to extract
			num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
			seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

		Returns:
			a dictionary of extracted items.
		"""
        data, pad_mask = self.get_sequence_from_demo(
            demo_id=demo_id,
            dataset_name=dataset_name,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def set_action_normalization_stats(self, action_normalization_stats):
        self.action_normalization_stats = action_normalization_stats

    def get_action_normalization_stats(self):
        return self.action_normalization_stats

    def get_dataset_sampler(self):
        """
		Return instance of torch.utils.data.Sampler or None. Allows
		for dataset to define custom sampling logic, such as
		re-weighting the probability of samples being drawn.
		See the `train` function in scripts/train.py, and torch
		`DataLoader` documentation, for more info.
		"""
        return None

