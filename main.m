clc;clear;
%% Step0: VAD и ХАРАКТЕРИСТИКИ ТЕСТОВОГО И ТРЕИН ДАННЫХ
nworkers = 8;
MainPath = 'D:\i-vectorsTest\';
WavFileTrain = strcat(MainPath,'Кайрат-4.wav');
[Ytrain,Fs] = audioread(WavFileTrain);
Ytrain1 = Ytrain(:,1);
FrsTrain=NN_test_VAD(Ytrain1,Fs);
WavFile = strcat(MainPath,'test.wav');
[Ytest,Fs] = audioread(WavFile);
Ytest1 = Ytest(:,1);
FrsTest=NN_test_VAD(Ytest1,Fs);

%% Step1: Считывание УФМ
UBMFile = strcat(MainPath,'\finalUBMonNN_1024.mat');
UBM_GMM = load(UBMFile, 'GMModel');
ubm = UBM_GMM.GMModel;

%% Step2: Обучение Т-матрицы полной изменчивости
tv_dim = 400; 
% niter  = 5;
TVFile = strcat(MainPath,'\FinalUbmData_1024-16.mat');
TV_data = load(TVFile, 'DataList');
% feaFiles=TV_data.DataList;
% stats = cell(length(feaFiles), 1);
% % feaFiles = C{1};
% parfor file = 1 : length(feaFiles)
%     [N, F] = compute_bw_stats(feaFiles{file}, ubm);
%     stats{file} = [N; F];
% end
% T = train_tv_space(stats, ubm, tv_dim, niter, nworkers);
load('TV.mat','T');
load('statsF.mat','stats')
%% Step3: Обучение Gaussian PLDA модели
lda_dim = 200;
nphi    = 200;
niter   = 10;
feaFiles=TV_data.DataList;
dev_ivs = zeros(tv_dim, length(feaFiles));
parfor file = 1 : length(feaFiles)
    dev_ivs(:, file) = extract_ivector(stats{file}, ubm, T);
end
% reduce the dimensionality with LDA
C{2}=1:length(feaFiles);
spk_labs = C{2};
V = lda(dev_ivs, spk_labs);
dev_ivs = V(:, 1 : lda_dim)' * dev_ivs;
%------------------------------------
plda = gplda_em(dev_ivs, spk_labs, nphi, niter);

%% Step4: Сравнение векторов модели и тестового образца
model_ivs1 = zeros(tv_dim, 1);
        [N, F] = compute_bw_stats(FrsTrain, ubm);
        model_ivs1(:, 1) = extract_ivector([N; F], ubm, T);
test_ivs = zeros(tv_dim, 1);
    [N, F] = compute_bw_stats(FrsTest, ubm);
    test_ivs(:, 1) = extract_ivector([N; F], ubm, T);
    
model_ivs1 = V(:, 1 : lda_dim)' * model_ivs1;
test_ivs = V(:, 1 : lda_dim)' * test_ivs;
%------------------------------------
scores = score_gplda_trials(plda, model_ivs1, test_ivs);
disp(scores)