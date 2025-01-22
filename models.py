import math
import torch
import itertools
import torchaudio
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from transformers import UniSpeechSatForXVector, WavLMForXVector




class Classic(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1,
                 groups:int=1,
                 bias:bool=False,
                 padding_mode:str='zeros',
                 *args, **kwargs) -> None:
        super().__init__()
        
        self.Convolution = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    
    def forward(self, INPUT):
        return self.Convolution(INPUT)

class Depthwise(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1,
                 groups:int=1,
                 bias:bool=True,
                 padding_mode:str='zeros',
                 *args, **kwargs) -> None:
        super().__init__()
        
        self.Convolution = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias, padding_mode),
            nn.Conv1d(in_channels, out_channels, 1, stride, padding, dilation, groups, bias, padding_mode)
        )
    
    def forward(self, INPUT):
        return self.Convolution(INPUT)

class Lowrank(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 rank:int=1,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1,
                 groups:int=1,
                 bias:bool=True,
                 padding_mode:str='zeros',
                 *args, **kwargs) -> None:
        super().__init__()
        
        self.Convolution = nn.Sequential(
             nn.Conv1d(in_channels, rank, kernel_size, stride, padding, dilation, groups, bias, padding_mode),
             nn.Conv1d(rank, out_channels, 1, stride, padding, dilation, groups, bias, padding_mode)
        )
    
    def forward(self, INPUT):
        return self.Convolution(INPUT)

_Convolutions_ = {
    "classic": Classic,
    "depthwise": Depthwise,
    "lowrank": Lowrank
}

_Normalizations_ = {
    "batchnorm": nn.BatchNorm1d,
    "none": nn.Identity
}

_Activations_ = {
    "relu": nn.ReLU,
    "none": nn.Identity
}

class CNA_(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 rank:int=1,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1,
                 groups:int=1,
                 bias:bool=True,
                 padding_mode:str='zeros',
                 convolution_type:str="classic",
                 activation_type:str="relu",
                 normalization_tpe:str="batchnorm",
                 *args, **kwargs) -> None:
        super().__init__()
        
        self.Convoution = _Convolutions_[convolution_type](in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=kernel_size,
                                                           rank=rank,
                                                           stride=stride,
                                                           padding=padding,
                                                           dilation=dilation,
                                                           groups=groups,
                                                           bias=bias,
                                                           padding_mode=padding_mode,)
        
        self.Normalization = _Normalizations_[normalization_tpe](out_channels)
        
        self.Activation = _Activations_[activation_type]()
        
    def forward(self, INPUT):
        return self.Activation(self.Normalization(self.Convoution(INPUT)))

class SqueezeExcite(nn.Module):
    def __init__(self,
                 in_channels:int,
                 *args, **kwargs) -> None:
        super().__init__()
        
        self.SE = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                CNA_(in_channels=in_channels,
                                     out_channels=int(in_channels/2),
                                     kernel_size=1,
                                     padding=0,
                                     normalization_tpe="batchnorm"),
                                nn.Conv1d(in_channels=int(in_channels/2),
                                          out_channels=in_channels,
                                          kernel_size=1,
                                          padding=0),
                                nn.Sigmoid())
    def forward(self, INPUT):
        return INPUT * self.SE(INPUT)
    
class ChannelAggregation(nn.Module):
    def __init__(self,
                 in_channels:int,
                 se_channel:int,
                 kernel_size:int,
                 dilation:int,
                 scale:int,
                 *args, **kwargs) -> None:
        super().__init__()
        
        self.width = int(math.floor(se_channel/scale))
        self.LAYER_1 = CNA_(in_channels=in_channels,
                            out_channels=self.width*scale,
                            kernel_size=1)
        self.number = scale - 1
        padding_ = math.floor(kernel_size/2)*dilation
        
        self.LAYER_N = nn.ModuleList([CNA_(in_channels=self.width,
                                           out_channels=self.width,
                                           kernel_size=kernel_size,
                                           dilation=dilation,
                                           padding=padding_) for _ in  range(self.number)])
        
        self.LAYER_2 = CNA_(in_channels=self.width*scale,
                            out_channels=se_channel,
                            kernel_size=1)

        self.LAYER_SE = SqueezeExcite(se_channel)
        
    def forward(self, INPUT):
        RES = INPUT
        O = self.LAYER_1(INPUT)
        
        S = torch.split(O, self.width, 1)
        for i in range(self.number):
            if i == 0:
                S_ = S[i]
            else:
                S_ = S_ + S[i]
            S_ = self.LAYER_N[i](S_)
            if i == 0:
                O = S_
            else:
                O = torch.cat((O, S_), 1)
        O = torch.cat((O, S[self.number]), 1)
        O = self.LAYER_2(O)
        O = self.LAYER_SE(O)
        return O.add(RES)

class PreEMPHASIS(nn.Module):
    def __init__(self,
                 coefficient:float = 0.97,
                 *args, **kwargs):
        super().__init__()
        
        self.coefficient = coefficient
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coefficient, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, INPUT: torch.tensor) -> torch.tensor:
        if len(INPUT.shape) == 1:
            return F.conv1d(INPUT.unsqueeze(0), self.flipped_filter).squeeze(0)
        elif len(INPUT.shape) == 2:
            return F.conv1d(INPUT.unsqueeze(1), self.flipped_filter).squeeze()
        elif len(INPUT.shape) == 3:
            return F.conv1d(INPUT, self.flipped_filter)
             
class FilterBankAUGMENT(nn.Module):
    def __init__(self,
                 freq_mask_width:Tuple=(0, 8),
                 time_mask_width:Tuple=(0, 10),
                 *args, **kwargs) -> None: 
        super().__init__()
        
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
       

    def mask_along_axis(self, x:torch.tensor, dim:int):
        original_size = x.shape
        
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, INPUT:torch.Tensor) -> torch.Tensor:
        INPUT = self.mask_along_axis(INPUT, dim=2)
        INPUT = self.mask_along_axis(INPUT, dim=1)
        return INPUT

class Frontend(nn.Module):
    def __init__(self,
                 augmentation:bool=False,
                 *args, **kwargs) -> None:
        super().__init__()        
        
        self.augmention = augmentation
        self.PREEMPHASIS = PreEMPHASIS()
        self.MELSPEC = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                            n_fft=512,
                                                            win_length=400,
                                                            hop_length=160,
                                                            f_min=20,
                                                            f_max=7600,
                                                            window_fn=torch.hamming_window,
                                                            n_mels=80)

    def forward(self, INPUT):
        with torch.no_grad():
            INPUT = self.PREEMPHASIS(INPUT)
            
            INPUT = self.MELSPEC(INPUT.to(torch.float32))
            INPUT += 1e-6
            INPUT = INPUT.log()
            INPUT = INPUT - torch.mean(INPUT, dim=-1, keepdim=True)
            if self.augmention:
                INPUT = FilterBankAUGMENT().forward(INPUT.squeeze(1))
            return INPUT
             
class ECAPA_TDNN(nn.Module):
    def __init__(self,
                 C:int=1024,
                 embedding:int=512,
                 frontend:str="spectrogram",
                 number_of_blocks:int=3,
                 training:bool=True,
                 *args, **kwargs) -> None:
        super().__init__()
        
        if frontend == "spectrogram":
            self.FRONTEND = Frontend(augmentation=training)

        elif frontend == "rawaudio":
            self.FRONTEND = nn.Conv1d(in_channels=1,
                                      out_channels=80,
                                      kernel_size=400,
                                      stride=160)
        
        self.RELU = nn.ReLU()
            
        self.LAYER_1 = CNA_(in_channels=80, 
                            out_channels=C,
                            kernel_size=5,
                            stride=1,
                            padding=2)
        self.BLOCKS = nn.ModuleList([ChannelAggregation(in_channels=C,
                                                        se_channel=C,
                                                        kernel_size=3,
                                                        dilation=2+i,
                                                        scale=8) for i in range(number_of_blocks)])
        
        self.LAYER_2 = nn.Conv1d(in_channels=number_of_blocks*C,
                                 out_channels=1536,
                                 kernel_size=1)
        
        self.ATTENTION = nn.Sequential(CNA_(in_channels=4608,
                                            out_channels=256,
                                            kernel_size=1),
                                       nn.Tanh(),
                                       nn.Conv1d(in_channels=256, 
                                                 out_channels=1536,
                                                 kernel_size=1),
                                       nn.Softmax(dim=2))
        self.LAYER_3 = nn.BatchNorm1d(num_features=3072)
        self.LAYER_4 = nn.Linear(in_features=3072, out_features=embedding)
        self.LAYER_5 = nn.BatchNorm1d(embedding)
        
    def forward(self, INPUT, *args, **kwargs):
        O = self.FRONTEND(INPUT)
        O = self.LAYER_1(O)

        O_ = []
        O1 = O
        for B in self.BLOCKS:            
            O1 = B(O1)
            O_.append(O1)
            O1 += O1
        
        O = self.RELU(self.LAYER_2(torch.cat(O_, dim=1)))
        T = O.size()[-1]
        GLOBAL_O = torch.cat((O,
                              torch.mean(O, dim=2, keepdim=True).repeat(1,1,T),
                              torch.sqrt(torch.var(O, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1,1,T)),
                             dim=1)
        
        W = self.ATTENTION(GLOBAL_O)
        MU = torch.sum(O*W, dim=2)
        STD = torch.sqrt((torch.sum((O**2)*W, dim=2) - MU**2 ).clamp(min=1e-4))
        
        O = torch.cat((MU, STD), dim=1)
        O = self.LAYER_3(O)
        O = self.LAYER_4(O)
        O = self.LAYER_5(O)
        return O

class SileroVAD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Code refrenced from "get_speech_timestamps" in torch.hub.load()
        """
        
        self.VAD, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                     model="silero_vad",
                                     force_reload=True,
                                     trust_repo=True)
        
        self.threshold = 0.5
        self.min_speech_duration_ms = 250
        self.max_speech_duration_s = float('inf')
        self.min_silence_duration_ms = 100
        self.window_size_samples = 512
        self.speech_pad_ms = 30
        self.sampling_rate = kwargs.get("sample_rate", 16000)
        
        self.step = 1
        self.min_speech_samples = self.sampling_rate * self.min_speech_duration_ms / 1000
        self.speech_pad_samples = self.sampling_rate * self.speech_pad_ms / 1000
        self.max_speech_samples = self.sampling_rate * self.max_speech_duration_s - self.window_size_samples - 2 * self.speech_pad_samples
        self.min_silence_samples = self.sampling_rate * self.min_silence_duration_ms / 1000
        self.min_silence_samples_at_max_speech = self.sampling_rate * 98 / 1000
    
    def forward(self, INPUT):
        self.VAD.reset_states()
        audio_length_samples = len(INPUT)
        
        with torch.no_grad():
            
            speech_probs = []
            for current_start_sample in range(0, audio_length_samples, self.window_size_samples):
                chunk = INPUT[current_start_sample: current_start_sample + self.window_size_samples]
                if len(chunk) < self.window_size_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, int(self.window_size_samples - len(chunk))))
                speech_probs.append(self.VAD(chunk, self.sampling_rate).item())
                
            triggered = False
            speeches = []
            current_speech = {}
            neg_threshold = self.threshold - 0.15
            temp_end = 0 
            prev_end = next_start = 0 

            for i, speech_prob in enumerate(speech_probs):
                if (speech_prob >= self.threshold) and temp_end:
                    temp_end = 0
                    if next_start < prev_end:
                        next_start = self.window_size_samples * i

                if (speech_prob >= self.threshold) and not triggered:
                    triggered = True
                    current_speech['start'] = self.window_size_samples * i
                    continue

                if triggered and (self.window_size_samples * i) - current_speech['start'] > self.max_speech_samples:
                    if prev_end:
                        current_speech['end'] = prev_end
                        speeches.append(current_speech)
                        current_speech = {}
                        if next_start < prev_end: # previously reached silence (< neg_thres) and is still not speech (< thres)
                            triggered = False
                        else:
                            current_speech['start'] = next_start
                        prev_end = next_start = temp_end = 0
                    else:
                        current_speech['end'] = self.window_size_samples * i
                        speeches.append(current_speech)
                        current_speech = {}
                        prev_end = next_start = temp_end = 0
                        triggered = False
                        continue

                if (speech_prob < neg_threshold) and triggered:
                    if not temp_end:
                        temp_end = self.window_size_samples * i
                    if ((self.window_size_samples * i) - temp_end) > self.min_silence_samples_at_max_speech : # condition to avoid cutting in very short silence
                        prev_end = temp_end
                    if (self.window_size_samples * i) - temp_end < self.min_silence_samples:
                        continue
                    else:
                        current_speech['end'] = temp_end
                        if (current_speech['end'] - current_speech['start']) > self.min_speech_samples:
                            speeches.append(current_speech)
                        current_speech = {}
                        prev_end = next_start = temp_end = 0
                        triggered = False
                        continue
            
            if current_speech and (audio_length_samples - current_speech['start']) > self.min_speech_samples:
                current_speech['end'] = audio_length_samples
                speeches.append(current_speech)

            for i, speech in enumerate(speeches):
                if i == 0:
                    speech['start'] = int(max(0, speech['start'] - self.speech_pad_samples))
                if i != len(speeches) - 1:
                    silence_duration = speeches[i+1]['start'] - speech['end']
                    if silence_duration < 2 * self.speech_pad_samples:
                        speech['end'] += int(silence_duration // 2)
                        speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
                    else:
                        speech['end'] = int(min(audio_length_samples, speech['end'] + self.speech_pad_samples))
                        speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - self.speech_pad_samples))
                else:
                    speech['end'] = int(min(audio_length_samples, speech['end'] + self.speech_pad_samples))


            if self.step > 1:
                for speech_dict in speeches:
                    speech_dict['start'] *= self.step
                    speech_dict['end'] *= self.step
                    
            return [INPUT[i["start"]:i["end"]] for i in speeches]

class PredictionModel(nn.Module):
    def __init__(self,
                 number_classes,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.layer = nn.Sequential(
            nn.Linear(512, int(512*0.50)),
            nn.ReLU(),
            nn.Linear(int(512*0.50), int(512*0.25)),
            nn.ReLU(),
            nn.Linear(int(512*0.25), number_classes)
        )
    
    def forward(self, INPUT):
        return self.layer(INPUT)

class PyAnnote_Segmentation_V3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.pipeline = VoiceActivityDetection(
            segmentation=Model.from_pretrained("pyannote/segmentation-3.0", 
                                               use_auth_token="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")
        )
        HYPER_PARAMETERS = {
        "min_duration_on": 0.0,         # remove speech regions shorter than that many seconds.
        "min_duration_off": 0.0         # fill non-speech regions shorter than that many seconds.
        }
        self.pipeline.instantiate(HYPER_PARAMETERS)
    
    def forward(self, INPUT, SR):
        vad = self.pipeline(INPUT)
        return [(int(j[0]*SR), int(j[1]*SR)) for j in [(i.start, i.end) for i in vad.itersegments()]]
    
class UniSpeechPretrained(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = UniSpeechSatForXVector.from_pretrained('microsoft/unispeech-sat-base-plus-sv')
         
    def forward(self, INPUT):
        return self.model(INPUT).embeddings

class WavLMPretrained(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
    
    def forward(self, INPUT):
        return self.model(INPUT).embeddings

class ClusterLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        dist = kwargs.get("distance", "euclidean")

        if dist == "euclidean":
            self.distance = nn.PairwiseDistance(keepdim=True)
        elif dist == "cosine":
            self.distance = nn.CosineSimilarity(dim=0)

    def forward(self, logits, labels):
        uniq_classes = [
            torch.index_select(logits,0,(labels==j).nonzero().squeeze(1)) for j in labels.unique()
        ]
 
        interclass_loss = torch.tensor(
            list(
                itertools.chain.from_iterable(
                    [[self.distance(uniq_classes[i].mean(dim=0), uniq_classes[j].mean(dim=0)) for j in range(len(uniq_classes)) if i!=j] for i in range(len(uniq_classes))]
                ))
        ).sum()
        
        intraclass_loss = torch.tensor(
            list(
                itertools.chain.from_iterable(
                    [[self.distance(uniq_classes[i][j], uniq_classes[i].mean(dim=0)) for j in range(uniq_classes[i].shape[0])] for i in range(len(uniq_classes))]
                    )
                )    
        ).sum()        
        return intraclass_loss + (-interclass_loss)

class Cluster_classification_Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        self.cluster_loss = ClusterLoss(**kwargs)
        self.classfication_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        return self.cluster_loss(logits, labels) + self.classfication_loss(logits, labels)