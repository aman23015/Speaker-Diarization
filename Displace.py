import re
import sys
sys.path.append("./"), sys.path.append("../")

import argparse
import itertools
from models import *
from utilities import *
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import MiniBatchSparsePCA

P = argparse.ArgumentParser()

P.add_argument("gpu", type=int, default=0)
P.add_argument("epochs", type=int, default=120)
P.add_argument("batch_size", type=int, default=512)
P.add_argument("optimizer", type=str, default="adam")
P.add_argument("augment_percent", type=float, default=0.25)
P.add_argument("distance_metric", type=str, default="cosine")
P.add_argument("window_overlap", type=str, default="0.5_0.125")
P.add_argument("expeirment_name", type=str, default="diarization")
A = P.parse_args()

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
# Parameters & Paths
dev_data_directory="/home/hiddencloud/SERVER_DATASETS/DATASET_DISPLACE/2024/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
eval_data_directory="/home/hiddencloud/SERVER_DATASETS/DATASET_DISPLACE/2024/Displace2024_eval_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD"
dev_rttm_path="/home/hiddencloud/SERVER_DATASETS/DATASET_DISPLACE/2024/Displace2024_dev_labels_supervised/Labels/Track1_SD"
rirs_noises_dataset = "/home/hiddencloud/SERVER_DATASETS/DATASET_RIRS_NOISES/RIRS_NOISES/simulated_rirs/largeroom"
LOGPATH = "/home/hiddencloud/AMAN_MT23015"

cuda = A.gpu
aug_percent=A.augment_percent
dev_der_after_epoch = 2
window_overlap = A.window_overlap
dev_total_audios = len([i for i in sorted(os.listdir(dev_data_directory))])
eval_total_audios = len([i for i in sorted(os.listdir(eval_data_directory))])

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##

# Dataset

def metadata():
    dev_audios = sorted([os.path.join(dev_data_directory, i) for i in os.listdir(dev_data_directory)])
    eval_audios = sorted([os.path.join(eval_data_directory, i) for i in os.listdir(eval_data_directory)])
     
    dev_rttms = sorted([os.path.join(dev_rttm_path, i) for i in sorted(os.listdir(dev_rttm_path))])
    return dev_audios, dev_rttms, eval_audios

DevAudios, DevRTTMS, EvalAudios = metadata()

class ChunkedData(torch.utils.data.IterableDataset):
    def __init__(self, audio_path, rttm_path, augmentation=False):
        super().__init__()
        self.augmentation = augmentation
        self.audio, self.sr = Audio().load(audio_path, audio_duration="full")

        self.labels = rttm_reader(path=rttm_path) if rttm_path != None else None
        self.window = 1.5
        self.overlap = 0.5

    def rttm_to_labels(self,):
        speaker_map ={"S1":0, "S2":1, "S3":2, "S4":3, "S5":4, "NA":5}
        
        if self.labels == None:
            return None
        
        speaker = []
        for idx, data in enumerate(self.labels):
            start, duration = list(data[idx].values())[0][0], list(data[idx].values())[0][1]

            if idx == 0:
                if start != 0:
                    speaker.append((speaker_map[list(data[idx].keys())[0]], 0*self.sr, int(start*self.sr)))
                else:
                    speaker.append((speaker_map[list(data[idx].keys())[0]], 0*self.sr, int(self.sr*duration)))
            else: speaker.append((speaker_map[list(data[idx].keys())[0]], int(start*self.sr), int((start+duration)*self.sr)))


        indexes = [(k,k+int(self.sr*self.window)) for k in range(0, self.audio.shape[-1]-int(self.sr*self.window)+1, int(self.sr*self.overlap))]
        self.segments = indexes
        labels = []
        for idx in indexes:
            s = []
            for spk in speaker:
                if idx[0] <= spk[2]:
                    r1, r2 = range(idx[0], idx[1]), range(spk[1], spk[2])
                    if len(range(max(r1[0], r2[0]), min(r1[-1], r2[-1])+1)) != 0:
                        s.append(spk[0])
                    else:pass
                else:pass
            if len(s) == 1:
                labels.append(s[0])
            elif len(s) == 0:
                labels.append(5)
            elif len(s) > 1:
                labels.append(int(sum(s)/len(s)))     
        return torch.tensor(labels)    

    def augmented_data(self, ):
        rirs_noise_list = []
        for i in os.listdir(rirs_noises_dataset):
            if "Room" in i:
                for j in os.listdir(os.path.join(rirs_noises_dataset, i)):
                    rirs_noise_list.append(os.path.join(rirs_noises_dataset,i,j))
        
        def add_rir(audio, rir_audio_path, max_len):
            rir, _ = torchaudio.load(rir_audio_path)
            rir = rir[0][:] / torch.sqrt(torch.sum(rir**2))
            return torch.tensor(scisignal.convolve(audio, rir, mode='full')[:max_len])
        
        A = self.audio[0].unfold(0, size=int(self.sr*self.window), step=int(self.sr*self.overlap))
        indices_to_augment = list(set([random.randint(0, A.shape[0]-1) for _ in range(int(aug_percent*A.shape[0]))]))
        for i in indices_to_augment:
            A[i] = add_rir(audio=A[i],
                           rir_audio_path=random.choice(rirs_noise_list),
                           max_len=A.shape[-1]) 
        return A
    
    def __iter__(self,):
        if self.augmentation:
            A = self.augmented_data()
        else:
            A = self.audio[0].unfold(0, size=int(self.sr*self.window), step=int(self.sr*self.overlap))
        
        if self.labels == None:
            for i in self.segments:
                yield A[i], torch.tensor(i)
        else:
            L = self.rttm_to_labels()
            for i in range(len(self.segments)):
                yield A[i], L[i]


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##

# Speaker Embedding Model, Prediction Model, Criterion

embedding_model = ECAPA_TDNN()
embedding_model.load_state_dict(
    state_dict=torch.load("/home/hiddencloud/AMAN_MT23015/ARCHIVES/Sub_Root/Epoch_120.pth")["Architecture"]
)
prediction_model = PredictionModel(number_classes=6)

embedding_model.eval()
prediction_model.train()

# Criterion
criterion = nn.CrossEntropyLoss()

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##           

# Optimizer
if A.optimizer == "adam":
    optimizer = torch.optim.Adam(params=list(embedding_model.parameters())+list(prediction_model.parameters()),
                                lr=0.0003,
                                weight_decay=1e-3)
elif A.optimizer == "sgd":
    optimizer = torch.optim.SGD(params=list(embedding_model.parameters())+list(prediction_model.parameters()),
                                lr=0.0003,
                                momentum=0.9,
                                weight_decay=1e-3)

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##     

class Diarization:
    def __init__(self,
                 embedding_architecture,
                 prediction_architecture,
                 criterion,
                 optimizer,
                 ) -> None:
        self.embedding_net = embedding_architecture
        self.prediction_net = prediction_architecture
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.logger = ChalkBoard(exp_name=A.expeirment_name, path=LOGPATH)
        self.device = torch.device(f"cuda:{cuda}") if cuda in range(0, torch.cuda.device_count()) else torch.device("cpu")
    
        self.embedding_net.to(self.device)
        self.prediction_net.to(self.device)

    def audio_dataloader(self, audio_path, rttm_path, eval=False, augmentation=False):
        if not eval:
            return DataLoader(dataset=ChunkedData(audio_path=audio_path,
                                                  rttm_path=rttm_path,
                                                  augmentation=augmentation),
                              batch_size=A.batch_size,
                              drop_last=False,
                              num_workers=2,
                              )
        else:
            return DataLoader(dataset=ChunkedData(audio_path=audio_path,
                                                  rttm_path=None),
                              batch_size=A.batch_size,
                              drop_last=False,
                              num_workers=2,
                              )
        
        
    def rttm_spk_st_dur(self,label,segment):

        # File containing speaker labels per segments
        seg2label = {}
        for line in label:
            seg = "{} {} {}".format(
                line[0],
                line[1],
                line[2]
            )
            label = line[-1]
            seg2label[seg] = label

        # Segment file 

        reco2segs = {}
        for i in segment:
            reco,start,end = i[0],i[1],i[2]
            seg = "{} {} {}".format(
                reco,
                start,
                end
            )
            if reco in reco2segs:
                reco2segs[reco] = reco2segs[reco] + " " + str(round(start,3)) + "," + str(round(end,3)) + "," + seg2label[seg]
            else :
                reco2segs[reco] = reco + " " + str(round(start,3)) + ","+str(round(end,3)) + "," + seg2label[seg]  
                
        # cut up overlapping segments so they are contigous
        contiguous_segs = []
        for reco in sorted(reco2segs):
            segs = reco2segs[reco].strip().split()
            new_segs = ""
            for i in range(1,len(segs)-1):
                start,end,label = segs[i].split(',')
                next_start,next_end,next_label = segs[i+1].split(',')
                if float(end) > float(next_start):
                    done = False
                    avg = str((float(next_start)+float(end))/2.0)
                    segs[i+1] = ','.join([avg,next_end,label])
                    new_segs += " "+start+","+avg+","+label
                else:
                    new_segs += " "+ start+","+end+","+label
            start,end,label = segs[-1].split(',')
            new_segs += " "+ start +","+end+","+label  
            contiguous_segs.append(reco + new_segs)

        # Merge contiguous segments of the same label
        merged_segs = []
        for reco_line in contiguous_segs:
            segs = reco_line.strip().split()
            reco = segs[0]
            new_segs = ""
            for i in range(1,len(segs)-1):
                start,end,label = segs[i].split(',')
                next_start,next_end,next_label = segs[i+1].split(',')
                if float(end) == float(next_start) and label == next_label:
                    segs[i+1] = ','.join([start,next_end,next_label])
                else:
                    new_segs += " "+ start +","+end+","+label
            start,end,label = segs[-1].split(',')
            new_segs += " " + start + ","+end+","+label
            merged_segs.append(reco+new_segs)

        # Making Rttm and hypths_data
        rttm_data, hypths_data = [], []
        for reco_line in merged_segs:
            segs = reco_line.strip().split()
            reco = segs[0]
            
            for i in range(1,len(segs)):
                start,end,label = segs[i].strip().split(',')
                hypths_data.append((label,start,end))
                rttm_data.append(
                    "SPEAKER {} 1 {} {} <NA> <NA> {} <NA> <NA>".format(
                        reco,
                        start,
                        end,
                        label
                    )
                )
        return hypths_data,rttm_data


    def minibatch_process(self, batch, eval=False):
        optimizer.zero_grad()
        
        if eval:
            data = batch
            data = data.to(self.device)
            logits = self.embedding_net(data)
            predcs = self.prediction_net(logits)
            return F.softmax(predcs, dim=1).argmax(1)

        else:
            data, label = batch
            data, label = data.to(self.device), label.to(self.device)
            print(data.shape)
            input("wait")
            logits = self.embedding_net(data)
            predcs = self.prediction_net(logits)
            loss = self.criterion(predcs, label)
            loss.backward()
            optimizer.step()
            accuracy = label.eq(F.softmax(predcs, dim=1).argmax(1)).sum().div(A.batch_size).mul(100)
            return loss.item(), accuracy.item()        

    def train(self,):
        torch.cuda.empty_cache()
        best_der = [77.00]
        for epoch in range(A.epochs):

            e_loss, e_acc, e_der = 0, 0, 0
            
            for d_idx in range(len(DevAudios)):
                aud_loss, aud_acc, mb_idx = 0, 0, 0
                for minibatch in self.audio_dataloader(audio_path=DevAudios[d_idx],
                                                       rttm_path=DevRTTMS[d_idx],
                                                       augmentation=True):
                    mb_idx += 1
                    loss, accuracy = self.minibatch_process(batch=minibatch)
                    aud_loss += loss
                    aud_acc += accuracy

                print(
                    "Epoch: {}, Audio: {}, [Loss: {}, Accuracy: {}]".format(
                        epoch,
                        d_idx,
                        round(aud_loss/mb_idx, 4),
                        round(aud_acc/mb_idx, 4),
                        ),
                    end="\r"
                )
                e_loss += aud_loss/mb_idx
                e_acc += aud_acc/mb_idx   

            # if epoch%dev_der_after_epoch == 0:
            #     for d_idx in range(len(DevAudios)):
            #         # ID = DevSegments[d_idx].split("_")[-3]
            #         var = DevAudios[d_idx]
            #         match = re.search(r'/([^/]+)\.wav$', var)
            #         ID = match.group(1)
            #         dev_labels = []
            #         for minibatch in self.audio_dataloader(audio_path=DevAudios[d_idx],
            #                                                rttm_path=None
            #                                                ):
            #             predcs = self.minibatch_process(batch=minibatch, eval=True)
            #             dev_labels.append(predcs.tolist())
                    
            #         dev_labels = list(itertools.chain.from_iterable(dev_labels))
            #         # dev_segments = list(itertools.chain.from_iterable(dev_segments))

            #         hypths_data, rttm_data = self.rttm_spk_st_dur(
            #             labels=dev_labels,
            #             # segments=dev_segments,
            #             audio_path=DevAudios[d_idx],
            #             # segment_path=DevSegments[d_idx]
            #         )

            #         ref_data = create_ref_rttm(DevRTTMS[d_idx])
            #         e_der += DER().compute(ref=ref_data, hyp=hypths_data)
                    
            #         with open(os.path.join(self.logger.rttm_path, f"{ID}_SPEAKER_sys.rttm"), "a") as FILE:FILE.writelines([i+"\n" for i in rttm_data]);FILE.close()                     

            # details = "Epoch: {}, [Loss: {}, Accuracy: {}, DER: {}]".format(
            #     epoch,
            #     round(e_loss/dev_total_audios, 4),
            #     round(e_acc/dev_total_audios, 4),
            #     round(e_der/dev_total_audios, 4)
            #     )
            # print(details)
            # self.logger.scribe(details)

            # if round(e_der/dev_total_audios, 4) < best_der[0]:
            #     torch.save(obj={"embedding": self.embedding_net.state_dict(),
            #                     "prediction": self.prediction_net.state_dict()},
            #             f=os.path.join(self.logger.exp_path, f"checkpoint.pth"))
            #     best_der[0] = round(e_der/dev_total_audios, 4)
            # else:pass

    def clustering(self,):
        self.embedding_net.eval()

        e_der = 0
        score = [0.0]
        pca_components = 7
        class_clusters = 5

        for iteration in range(A.epochs):
            
            transformer = MiniBatchSparsePCA(n_components=pca_components, batch_size=128, max_iter=24, random_state=0)    
            KMeansModel = MiniBatchKMeans(n_clusters=class_clusters, verbose=1, max_iter=333)
            
            iter_score, iter_accuracy = 0, 0
            for d_idx in range(len(DevAudios)):
                    mb_score, mb_acc, mb = 0, 0, 0
                    dataloader = self.audio_dataloader(audio_path=DevAudios[d_idx],
                                                       rttm_path=DevRTTMS[d_idx],
                                                       augmentation=True)
                    for minibatch in dataloader:
                        mb += 1
                        data, label, _ = minibatch
                        logits = F.normalize(input=self.embedding_net(data.to(self.device)), p=2.0).detach().cpu().numpy()
                        logits = transformer.fit_transform(logits)
                        KMeansModel.partial_fit(logits)
                        
                        predictions = KMeansModel.predict(logits)
                        mb_score += silhouette_score(data=logits, labels=label.numpy())
                        mb_acc += label.eq(torch.tensor(predictions)).sum().div(predictions.shape[0]).mul(100).item()
                        print("Audio: {}, Minibatch: {}, Score: {}".format(
                            d_idx, mb, mb_score
                        ))

                    iter_score += mb_score/mb
                    iter_accuracy += mb_acc/mb
                    self.logger.scribe(
                        "Iteration: {}, [Audio: {}, Score: {}, Accuracy: {}]".format(
                            iteration,
                            d_idx,
                            mb_score/mb,
                            mb_acc/mb
                        )
                    )   
            if iter_score/dev_total_audios < 0:
                print("Creating DEV Rttms")
                pca_components += random.randint(-2,2)
                class_clusters += random.randint(-2,2)
            else:
                pass   

            for d1_idx in range(len(DevAudios)):
                var = DevAudios[d1_idx]
                match = re.search(r'/([^/]+)\.wav$', var)
                ID = match.group(1)
                dev_labels, dev_segments = [], []
                dataloader = self.audio_dataloader(audio_path=DevAudios[d1_idx],
                                                   rttm_path=DevRTTMS[d1_idx])
                for d1_minibatch in dataloader:
                    d1_data, _, d1_segment = d1_minibatch
                    logits = F.normalize(input=self.embedding_net(d1_data.to(self.device)), p=2.0).detach().cpu().numpy()
                    logits = transformer.fit_transform(logits)
                    d1_predcs = KMeansModel.predict(logits)
                    dev_labels.append(list(d1_predcs)), dev_segments.append(d1_segment.tolist())

                dev_labels = list(itertools.chain.from_iterable(dev_labels))
                dev_segments = list(itertools.chain.from_iterable(dev_segments))

                hypths_data, rttm_data = self.rttm_spk_st_dur(
                    labels=dev_labels,
                    # segments=dev_segments,
                    audio_path=DevAudios[d1_idx],
                    # segment_path=DevSegments[d1_idx]
                )

                ref_data = create_ref_rttm(DevRTTMS[d1_idx])
                e_der += DER().compute(ref=ref_data, hyp=hypths_data)
                
                with open(os.path.join(self.logger.rttm_path, f"{ID}_SPEAKER_sys.rttm"), "a") as FILE:FILE.writelines([i+"\n" for i in rttm_data]);FILE.close()



if __name__ == "__main__":
    Trainer = Diarization(
        embedding_architecture=embedding_model,
        prediction_architecture=prediction_model,
        criterion=criterion,
        optimizer=optimizer
    )
    Trainer.train()
    # Trainer.evaluate()
    # Trainer.clustering()