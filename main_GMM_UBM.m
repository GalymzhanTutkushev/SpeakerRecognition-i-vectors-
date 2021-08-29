function scores = main_GMM_UBM()
%% Step0: VAD и ХАРАКТЕРИСТИКИ ТЕСТОВОГО И ТРЕИН ДАННЫХ
nworkers = 8;
MainPath = 'D:\i-vectorsTest\';
WavFileTrain = strcat(MainPath,'Кайрат-1.wav');
[Ytrain,Fs] = audioread(WavFileTrain);
Ytrain1 = Ytrain(:,1);
FrsTrain=NN_test_VAD(Ytrain1,Fs);
WavFile = strcat(MainPath,'Люда-2.wav');
[Ytest,Fs] = audioread(WavFile);
Ytest1 = Ytest(:,1);
FrsTest=NN_test_VAD(Ytest1,Fs);

%% Step1: Считывание УФМ
UBMFile = strcat(MainPath,'\finalUBMonNN_1024.mat');
UBM_GMM = load(UBMFile, 'GMModel');
ubm = UBM_GMM.GMModel;
   
%% Step2: Адаптация модели диктора через УФМ
nspks = 1;
map_tau = 10.0;
config = 'mwv';
gmm_models = cell(nspks, 1);   
spk_files={FrsTrain};
gmm_models{nspks} = mapAdapt(spk_files, ubm, map_tau, config);

%% Step3: Подсчет схожести аудиофайлов
test_files={FrsTest};
trials = [1, 1];
scores = score_gmm_trials(gmm_models, test_files, trials, ubm);
end