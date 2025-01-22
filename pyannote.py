from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

single_audio = "/home/hiddencloud/SERVER_DATASETS/DATASET_DISPLACE/2024/Displace2024_dev_audio_supervised/AUDIO_supervised/Track1_SD_Track2_LD/B007.wav"

diarization = pipeline(single_audio,num_speakers=4)

with open("/home/hiddencloud/AMAN_MT23015/New_Dis/audio.rttm","w") as rttm:
    diarization.write_rttm(rttm)