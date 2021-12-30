from model import ModelVad
from dataset import AVADataset
import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve
import json
import shutil
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def create_spec_dataset(speech_labels_paths: str, path_specs: str, delimiter: str = ','):
    '''
    Функция принимает на вход пути ко всем аудио датасета и преобразует их к спектрограммам
    Входные данные:
    speech_labels_paths: str - путь к датасету AVA формата
    path_specs: str - путь к папке, куда сохранять спектрограммы
    delimeter: str=',' - разделитель в датасете
    '''

    with open(speech_labels_paths) as f:
        speech_labels = f.readlines()
    speech_labels = [x.replace('\n', '') for x in speech_labels]

    if not os.path.exists(path_specs):
        os.makedirs(path_specs)

    i = 0
    labels = []
    sample_rate = 16000
    win_length = hop_length = n_fft = int(sample_rate / 100)
    step = int(0.32 * sample_rate)
    while i < len(speech_labels):
        path_to_audio = speech_labels[i].split(delimiter)[0]
        audio = path_to_audio.split('/')[-1].split('.')[0]
        try:
            waveform, _ = torchaudio.load(path_to_audio)
            print(path_to_audio)
        except:
            i += 1
            continue
        j = 0
        while True:
            speech_label_split = speech_labels[i].split(delimiter)
            second_start = float(speech_label_split[1])
            second_end = float(speech_label_split[2])
            label = speech_label_split[3]
            if label == 'NO_SPEECH':
                label = '0'
            else:
                label = '1'
            while (second_start * sample_rate < waveform.shape[1] - step) and (second_end - second_start > 0.32):
                specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, win_length=win_length,
                                                                n_mels=32, power=1, hop_length=hop_length,
                                                                n_fft=n_fft + 1)(
                    waveform[0][int(second_start * sample_rate):int(second_start * sample_rate) + step])
                data = specgram.log2().detach().numpy()
                if True in np.isinf(data):
                    j += 1
                    second_start += 0.32
                    continue
                np.save(path_specs + '/{0}_{1}'.format(audio, str(j)), data)
                labels.append([path_specs + '/{0}_{1}.npy'.format(audio, str(j)), label])
                j += 1
                second_start += 0.32
            i += 1
            if speech_labels[i].split(delimiter)[0] != path_to_audio:
                break
    return np.array(labels)

def create_test_dataset(data: str):
    '''
    Функция принимает на вход массив с указанием путей ко всем спектрограммам и их меткам
    и возвращает датафрейм с спутями к спектрограммам
    Входные данные:
    data: str - список путей к спектрограммам и их меток
    '''
    np.random.shuffle(data)
    df = pd.DataFrame(data)
    df = df.rename(columns={0: 'spec_paths', 1: 'labels'})
    m = df.loc[:, 'labels'] == '0'
    df.loc[m, 'labels'] = 0
    m = df.loc[:, 'labels'] != 0
    df.loc[m, 'labels'] = 1
    test_df = df.reset_index()
    return test_df

def test_result(model, test_data_loader: DataLoader):
    '''
    Функция принимает на вход модель и тестовый датасет и вычисляет по нему значение метрики и предсказанные
    значения
    Входные данные:
    model - обученная модель
    test_data_loader: DataLoader - тестовый датасет, приведенный к формату DataLoader
    '''
    model.eval()
    y_true = []
    y_proba = []
    for batch_idx, data in enumerate(test_data_loader):
        inputs = data[0]
        target = data[1]
        outputs = model(inputs)
        y_true.append(np.array(target.detach().numpy()))
        y_proba.append(np.array(outputs[:,0].detach().numpy()))

    y_true = np.concatenate(np.array(y_true), axis=0)
    y_proba = np.concatenate(np.array(y_proba), axis=0)
    return y_true, y_proba

def plot_PR_ROC(y_true, y_proba):
    '''
    Функция принимает на вход предсказзные моделью значения и истинные значения и строит PR кривую с f1 изолиниями
    Входные данные:
    y_true - истинные значения
    y_prob - предсказанные моделью значения
    '''
    figure = plt.figure(figsize = (16, 7))
    plt1 = figure.add_subplot(121)

    prec, recall, _ = precision_recall_curve(y_true, y_proba)
    plt1.plot(recall, prec, linewidth=2, color='r')

    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt1.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.5)

    plt1.set_xlim([0.0, 1.0])
    plt1.set_ylim([min(prec), 1.01])
    plt1.set_xlabel('Recall')
    plt1.set_ylabel('Precision')
    plt1.grid()

    plt2 = figure.add_subplot(122)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt2.set_xlim([0.0, 1.0])
    plt2.set_ylim([0.0, 1.0])
    plt2.plot(fpr, tpr)
    plt2.set_xlabel('False positive rate')
    plt2.set_ylabel('True positive rate')
    plt2.set_title(f'ROC curve. ROC_AUC = {roc_auc:.3f}')
    plt2.grid()

    plt.show()


if __name__ == '__main__':

    with open('config.json', 'r') as f:
        config = json.load(f)

    model = ModelVad()
    model.load_state_dict(torch.load(config['path_to_model']))

    labels = create_spec_dataset(config['speech_labels_path'], config['path_to_save_specs'], config['delimiter'])

    np.save(config['path_to_save_specs'] + '/labels', labels)

    test_df = create_test_dataset(labels)

    mean = -3.052366018295288
    std = 2.4621522426605225

    test_dataset = AVADataset(test_df, mean, std)

    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

    y_true, y_proba = test_result(model, test_data_loader)

    plot_PR_ROC(y_true, y_proba)

    if config['is_save_specs'] != '1':
        shutil.rmtree(config['path_to_save_specs'])