import pyaudio
import torchaudio
import torch
from model import ModelVad

#Размер пакета данных запсии в отсчетах
CHUNK = 5120
#Формат данных
FORMAT = pyaudio.paFloat32
#Количество каналов
CHANNELS = 1
#Частота записи
RATE = 16000
#Окно
WIN = int(RATE/100)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

#Значения среднего и СКО, вычисленные для спектрограмм на тестовом датасете
mean = -3.052366018295288
std = 2.4621522426605225

model = ModelVad()
model.load_state_dict(torch.load('./vad_model2_2.pth'))

while True:
    try:
        data = stream.read(CHUNK, exception_on_overflow = False)
    except:
        break
    data = torch.frombuffer(data, dtype=torch.float32).type(torch.FloatTensor)
    specgram = torchaudio.transforms.MelSpectrogram(sample_rate=RATE, win_length=WIN, n_mels=32, power=1,
                                                    hop_length=WIN, n_fft=WIN+1)(data)
    specgram = specgram.log2().detach().numpy()
    specgram = (specgram - mean) / std
    specgram = torch.FloatTensor(specgram.reshape(1, 1, 32, 32))
    out = model(specgram)
    print(float(out[0][0]))
    print('*'*int(float(out[0][0])*100))

stream.stop_stream()
stream.close()
p.terminate()