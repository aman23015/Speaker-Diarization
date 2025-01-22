import os
import math
import time
import torch
import numpy
import random
import librosa
import soundfile
import torchaudio
torchaudio.set_audio_backend("sox_io")
from scipy import optimize
import torch.nn.functional as F
import scipy.signal as scisignal
from typing import List, Dict, Tuple



class Audio:
    def __init__(self) -> None:
        pass
    
    def load(self,
             path:str,
             audio_duration=2,
             sample_rate:int=16000,
             backend:str="torchaudio",
             audio_normalization:bool=True,
             audio_concat_srategy:str="flip_n_join"):
        
        if backend not in ["torchaudio", "librosa", "soundfile"]:
            raise Exception(f"Only implemented for (torchaudio, librosa, soundfile)")
        if audio_concat_srategy not in ["flip_n_join", "repeat"]:
            raise Exception(f"Only implemented for (random_concat, flip_n_join, repeat)")
        

        if backend == "torchaudio":
            audio, sr = torchaudio.load(path)
        if backend == "librosa":
            audio, sr = librosa.load(path)
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        if backend == "soundfile":
            audio, sr = soundfile.read(path)
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            
            
        max_frames = audio_duration * sample_rate

        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sr, sample_rate, dtype=audio.dtype)
            audio = resampler(audio)
        else: pass
        
        if audio_duration == "full": 
            if audio_normalization:
                audio = torch.nn.functional.normalize(audio)
            else: pass
            return audio, sample_rate
        
        if audio.shape[1] < max_frames:
            if audio_concat_srategy == "flip_n_join":
                audio = torch.cat([audio, audio.flip((1,))]*int(max_frames/audio.shape[1]), dim=1)[0][:max_frames]
            if audio_concat_srategy == "repeat":
                audio = torch.tile(audio, (math.ceil(max_frames/audio.shape[1]),))[0][:max_frames]   
        else:
            start = random.randint(0, audio.shape[1]-max_frames + 1)
            audio = audio[0][start:start+max_frames]

        if audio.shape[-1] != max_frames:
            audio = F.pad(audio, (0, max_frames-audio.shape[-1]))
            
        if audio_normalization:
            if len(audio.shape) == 1:
                audio = torch.nn.functional.normalize(audio.unsqueeze(0))
            else:
                audio = torch.nn.functional.normalize(audio)

        return audio, sample_rate


    def __add_reverberate(self,
                        audio_signal:torch.Tensor,
                        rir_audio_path:str,
                        audio_duration=2) -> torch.Tensor:
        
        rir, sr = torchaudio.load(rir_audio_path)
        max_len = audio_duration * sr
        rir = rir / torch.sqrt(torch.sum(rir**2))
        return torch.tensor(scisignal.convolve(audio_signal, rir, mode='full')[:,:max_len])

    def __add_noise(self,
                  audio_signal:torch.Tensor,
                  noise:List[Dict],
                  audio_duration:int=2) -> torch.Tensor:
        
        clean_db = 10 * torch.log10(torch.mean(audio_signal ** 2) + 1e-4) 
        
        _all_noises = []
        for i in noise:
            _n, _ = self.load(i["path"], full=False, audio_duration=audio_duration)
            _n = numpy.stack([_n], axis=0)
            _n_db = 10 * numpy.log10(numpy.mean(_n ** 2) + 1e-4)
            _n_snr = i["snr"]
            _all_noises.append(numpy.sqrt(10 ** ((clean_db - _n_db - _n_snr) / 10)) * _n)
        _noise = numpy.sum(numpy.concatenate(_all_noises, axis=0), axis=0, keepdims=True)

        try:
            return audio_signal + _noise.squeeze(0)
        except Exception as _:
            if audio_signal.shape[1] > _noise.squeeze(0).shape[1]:
                return audio_signal + F.pad(torch.tensor(_noise.squeeze(0)), (0, audio_signal.shape[1] - _noise.squeeze(0).shape[1]), value=0)
            elif audio_signal.shape[1] < _noise.squeeze(0).shape[1]:
                return _noise.squeeze(0) + F.pad(audio_signal, (0, _noise.squeeze(0).shape[1] - audio_signal.shape[1]), value=0)
            
    def augment(self,
                audio_signal:torch.Tensor,
                musan_data_path:str,
                rirs_data_path:str,
                audio_duration:int=2,):
        
        NoiseSnR = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        NoiseCount = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        
        with open(musan_data_path) as F: musan_noise = F.readlines();F.close(); musan_noise = [i.replace("\n", "") for i in musan_noise]
        with open(rirs_data_path) as F: rirs_noise = F.readlines();F.close(); rirs_noise = [i.replace("\n", "") for i in rirs_noise]
        
        musan_noise_files = {"noise": [], "speech": [], "music": []}
        for i in musan_noise: musan_noise_files[i.split("/")[-3]].append(i)
    
        musan_noise = lambda noise_type: [
            {
                "path": i,
                "type": noise_type,
                "snr": random.uniform(NoiseSnR[noise_type][0], NoiseSnR[noise_type][1])
            } for i in random.sample(musan_noise_files[noise_type], random.randint(NoiseCount[noise_type][0], NoiseCount[noise_type][1]))
        ]
        
        aug_type = random.choice(["clean", "reverb", "speech", "music", "noise"])
        if aug_type == "clean":
            audio_signal = audio_signal
        if aug_type == "reverb":
            audio_signal = self.__add_reverberate(audio_signal, rir_audio_path=random.choice(rirs_noise), audio_duration=audio_duration)
        if aug_type == "speech":
            audio_signal = self.__add_noise(audio_signal, musan_noise(noise_type="speech"), audio_duration=audio_duration)
        if aug_type == "music":
            audio_signal = self.__add_noise(audio_signal, musan_noise(noise_type="music"), audio_duration=audio_duration)
        if aug_type == "noise":
            audio_signal = self.__add_noise(audio_signal, musan_noise(noise_type="noise"), audio_duration=audio_duration)
        if audio_signal.shape[0] != 1:
            return torch.mean(audio_signal, dim=0).unsqueeze(0)
        return audio_signal

def rttm_reader(style="displace24", path:str=None):
    if style == "displace24":
        with open(path, "r") as F: lines = F.readlines(); F.close()
        lines = ["-".join([j for j in i.replace("<NA>", "").replace("\n", "").split(" ")[3:] if j != " "]).replace("--", "").replace("---", "-") for i in lines]
        labels = [
            {idx: {i.split("-")[-1]: [float(i) for i in i.split("-")[:2]] }} for idx, i in enumerate(lines)
            ]
        return labels
    
class Timer:
    def __init__(self) -> None:
        self.start_time = None

    def Start(self,):
        self.start_time = time.time()
    
    def Stop(self):
        hours, remaining = divmod(time.time() - self.start_time, 3600)
        minutes, seconds = divmod(remaining, 60)
        o = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds).split(".")[0]
        self.start_time = None
        return o
    
class ChalkBoard:
    def __init__(self, exp_name, path) -> None:
        
        self.exp_path = os.path.join(path, f"Chkpts/{exp_name}")
        if not os.path.isdir(self.exp_path):
            os.makedirs(self.exp_path)
        
        self.board_filepath = os.path.join(self.exp_path, "board.txt")
        with open(self.board_filepath, "a") as F: F.write("Experiment Details >> "+"\n"); F.close()
        
        self.rttm_path = os.path.join(self.exp_path, "rttms")
        if not os.path.isdir(self.rttm_path):
            os.makedirs(self.rttm_path)
        
        self.evalrttm_path = os.path.join(self.exp_path, "eval_rttms")
        if not os.path.isdir(self.evalrttm_path):
            os.makedirs(self.evalrttm_path)
        
    def scribe(self, *args):
        with open(self.board_filepath, "a") as F:
            F.write(f">>  >> " + ", ".join([str(i) for i in args]) + "\n")
            F.close()

class IterableConverter(torch.utils.data.IterableDataset):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.length = len(self.dataset)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            i_start, i_end = 0, self.length
        else:
            i_start = 0
            per_worker = int(math.ceil((self.length - i_start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            i_start = i_start + worker_id * per_worker
            i_end = min(i_start + per_worker, self.length)
        
        for i in range(i_start, i_end):
            yield self.dataset.__getitem__(i)


def SpeakerSegmentFile(VADModel, audio_directory, set_, windows_list=[1.5, 0.7, 0.5], overlap_list=[0.5, 0.25, 0.125]):
    for audio_path in sorted(os.listdir(audio_directory)):
        
        audio, sr = torchaudio.load(os.path.join(audio_directory, audio_path))
        SpeakerSegments = VADModel.forward(os.path.join(audio_directory, audio_path), sr)

        for wd, od in zip(windows_list, overlap_list):        
            AudioSegments = torch.tensor(range(0, audio.shape[-1])).unfold(0, int(sr*wd), int(sr*od))
            D = []
            for i in SpeakerSegments:
                for j in range(AudioSegments.shape[0]):
                    if (i[0] in AudioSegments[j].tolist()) or (i[1] in AudioSegments[j].tolist()):
                        D.append(j)
                    else:
                        pass
            D = sorted(list(set(D)))
            
            with open("{}_{}_{}_{}.txt".format(set_,
                                               audio_path.replace(".wav", ""),
                                               str(wd),
                                               str(od)), "a") as FILE:
                FILE.writelines([str(i)+"\n" for i in D])
            FILE.close()
            

class DER:
    def __init__(self) -> None:
        # Overlap is not considered.
        pass
    
    def check_input(self, hyp):
        if not isinstance(hyp, list):
            raise TypeError("Input must be a list.")
        for element in hyp:
            if not isinstance(element, tuple):
                raise TypeError("Input must be a list of tuples.")
            if len(element) != 3:
                raise TypeError(
                    "Each tuple must have the elements: (speaker, start, end).")
            if not isinstance(element[0], str):
                raise TypeError("Speaker must be a string.")
            if not isinstance(element[1], float) or not isinstance(
                    element[2], float):
                raise TypeError("Start and end must be float numbers.")
            if element[1] > element[2]:
                raise ValueError("Start must not be larger than end.")

    def compute_total_length(self, hyp):
        total_length = 0.0
        for element in hyp:
            total_length += element[2] - element[1]
        return total_length


    def compute_intersection_length(self, A, B):
        max_start = max(A[1], B[1])
        min_end = min(A[2], B[2])
        return max(0.0, min_end - max_start)


    def compute_merged_total_length(self, ref, hyp):
        # Remove speaker label and merge.
        merged = [(element[1], element[2]) for element in (ref + hyp)]
        # Sort by start.
        merged = sorted(merged, key=lambda element: element[0])
        i = len(merged) - 2
        while i >= 0:
            if merged[i][1] >= merged[i + 1][0]:
                max_end = max(merged[i][1], merged[i + 1][1])
                merged[i] = (merged[i][0], max_end)
                del merged[i + 1]
                if i == len(merged) - 1:
                    i -= 1
            else:
                i -= 1
        total_length = 0.0
        for element in merged:
            total_length += element[1] - element[0]
        return total_length


    def build_speaker_index(self, hyp):
        speaker_set = sorted({element[0] for element in hyp})
        index = {speaker: i for i, speaker in enumerate(speaker_set)}
        return index


    def build_cost_matrix(self, ref, hyp):
        ref_index = self.build_speaker_index(ref)
        hyp_index = self.build_speaker_index(hyp)
        cost_matrix = numpy.zeros((len(ref_index), len(hyp_index)))
        for ref_element in ref:
            for hyp_element in hyp:
                i = ref_index[ref_element[0]]
                j = hyp_index[hyp_element[0]]
                cost_matrix[i, j] += self.compute_intersection_length(
                    ref_element, hyp_element)
        return cost_matrix


    def compute(self, ref, hyp):
        self.check_input(ref)
        self.check_input(hyp)
        ref_total_length = self.compute_total_length(ref)
        cost_matrix = self.build_cost_matrix(ref, hyp)
        row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
        optimal_match_overlap = cost_matrix[row_index, col_index].sum()
        union_total_length = self.compute_merged_total_length(ref, hyp)
        der = (union_total_length - optimal_match_overlap) / ref_total_length
        return der*100
    
    
def rttm_spk_st_dur(labels, segments, audio_path, segment_path):
    speaker_map = {"S1":0, "S2":1, "S3":2, "S4":3, "S5":4, "NA":5}
    ID = segment_path.split("_")[-3]
    window = float(segment_path.split("_")[-2])
    overlap = float(segment_path.split("_")[-1].replace(".txt", ""))
    audio, sr = torchaudio.load(audio_path)
    
    duration_segments = torch.tensor(range(audio.shape[-1])).unfold(0, size=int(sr*window), step=int(sr*overlap))
    changes = []
    for i, l in enumerate(labels[:-1]):
        if labels[i+1] != l: changes.append(i)
        else: pass
    changes.append(len(labels)-1)
    
    rttm_data, hypths_data = [], []
    for cidx, ch in enumerate(changes):
        if cidx == 0:
            c_start = duration_segments[segments[ch+1 if ch!=0 else ch]].min().div(sr)
            duration = (ch+1)*window
            label = labels[ch]
        else:
            c_start = duration_segments[segments[ch]].min().div(sr)
            duration = (changes[cidx]-changes[cidx-1])*window
            label = labels[ch]
        
        hypths_data.append(
            (list(speaker_map.keys())[list(speaker_map.values()).index(label)], round(c_start.item(), 3), round(c_start.item(), 3)+duration)
        )
        
        rttm_data.append(
            "SPEAKER {} 1 {} {} <NA> <NA> {} <NA> <NA>".format(
                ID,
                round(c_start.item(), 3),
                duration,
                list(speaker_map.keys())[list(speaker_map.values()).index(label)]
            )
        )
    return hypths_data, rttm_data

def create_ref_rttm(rttm_path):
    labels = rttm_reader(path=rttm_path)
    ref_rttms = [[*list(d[i].keys()), *list(*d[i].values())] for i, d in enumerate(labels)]
    for i in range(len(ref_rttms)):
        ref_rttms[i][-1] = ref_rttms[i][1]+ref_rttms[i][-1]
        ref_rttms[i] = tuple(ref_rttms[i])
    return ref_rttms

def silhouette_score(data, labels):
    n_samples, _ = data.shape
    
    # Calculate distances between all data points
    distances = numpy.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = numpy.linalg.norm(data[i] - data[j])
            distances[i, j] = distance
            distances[j, i] = distance

    # Calculate average intra-cluster distances
    intra_cluster_distances = numpy.zeros(n_samples)
    for i in range(n_samples):
        cluster_label = labels[i]
        cluster_indices = numpy.where(labels == cluster_label)[0]
        cluster_distances = distances[i, cluster_indices]
        intra_cluster_distances[i] = numpy.mean(cluster_distances) if len(cluster_distances) > 0 else 0

    # Calculate average nearest-cluster distances
    nearest_cluster_distances = numpy.zeros(n_samples)
    for i in range(n_samples):
        min_distance = numpy.inf
        for j in range(n_samples):
            if labels[i] != labels[j] and distances[i, j] < min_distance:
                min_distance = distances[i, j]
        nearest_cluster_distances[i] = min_distance

    # Calculate silhouette scores for all samples
    silhouette_scores = (nearest_cluster_distances - intra_cluster_distances) / numpy.max(
        numpy.vstack((intra_cluster_distances, nearest_cluster_distances)), axis=0
    )

    # Return the average silhouette score
    return numpy.mean(silhouette_scores)



# def get_labels(segs, n_frames):
#     """Return frame-wise labeling corresponding to a segmentation.

#     The resulting labeling is an an array whose ``i``-th entry provides the label
#     for frame ``i``, which can be one of the following integer values:

#     - 0:   indicates no speaker present (i.e., silence)
#     - 1:   indicates more than one speaker present (i.e., overlapped speech)
#     - n>1: integer id of the SOLE speaker present in frame

#     Speakers are assigned integer ids >1 based on their first turn in the
#     recording.

#     Parameters
#     ----------
#     segs : iterable of Segment
#         Recording segments.

#     n_frames : int
#         Length of recording in frames.

#     Returns
#     -------
#     ref : ndarray, (n_frames,)
#         Framewise speaker labels.
#     """
#     # Induce mapping from string speaker ids to integers > 1s.
#     n_speakers = 0
#     speaker_dict = {}
#     for seg in segs:
#         if seg.speaker_id in speaker_dict:
#             continue
#         n_speakers += 1
#         speaker_dict[seg.speaker_id] = n_speakers + 1

#     # Create reference frame labeling:
#     # - 0: non-speech
#     # - 1: overlapping speech
#     # - n>1: speaker n
#     # We use 0 to denote silence frames and 1 to denote overlapping frames.
#     ref = np.zeros(n_frames, dtype=np.int32)
#     for seg in segs:
#         # Integer id of speaker.
#         speaker_label = speaker_dict[seg.speaker_id]

#         # Assign this label to all frames in the segment that are not
#         # already assigned.
#         for ind in range(seg.onset, seg.offset+1):
#             if ref[ind] == speaker_label:
#                 # This shouldn't happen, but being paranoid in case the
#                 # initialization contains overlapping segments by same speaker.
#                 continue
#             elif ref[ind] == 0:
#                 label = speaker_label
#             else:
#                 # Overlapped speech.
#                 label = 1
#             ref[ind] = label

#     return ref



# def write_rttm_file(rttm_path, labels, channel=0, step=0.01, precision=2):
#     """Write RTTM file.

#     Parameters
#     ----------
#     rttm_path : Path
#         Path to output RTTM file.

#     labels : ndarray, (n_frames,)
#         Array of predicted speaker labels. See ``get_labels`` for explanation.

#     channel : int, optional
#         Channel (0-indexed) to output segments for.
#         (Default: 0)

#     step : float, optional
#         Duration in seconds between onsets of frames.
#         (Default: 0.01)

#     precision : int, optional
#         Output ``precision`` digits.
#         (Default: 2)
#     """
#     rttm_path = Path(rttm_path)

#     # Determine indices of onsets/offsets of speaker turns.
#     is_cp = np.diff(labels, n=1, prepend=-999, append=-999) != 0
#     cp_inds = np.nonzero(is_cp)[0]
#     bis = cp_inds[:-1]  # Last changepoint is "fake".
#     eis = cp_inds[1:] -1

#     # Write turns to RTTM.
#     with open(rttm_path, 'w') as f:
#         for bi, ei in zip(bis, eis):
#             label = labels[bi]
#             if label < 2:
#                 # Ignore non-speech and overlapped speech.
#                 continue
#             n_frames = ei - bi + 1
#             duration = n_frames*step
#             onset = bi*step
#             recording_id = rttm_path.stem
#             line = f'SPEAKER {recording_id} {channel} {onset:.{precision}f} {duration:.{precision}f} <NA> <NA> speaker{label} <NA> <NA>\n'
#             f.write(line)