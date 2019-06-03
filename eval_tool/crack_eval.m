model='FPHB';
fprintf('%s\n',model);
resDir='/media/fan/dv2/backup/fan_part3/home/fy/fan/hed/data/crack/testresults/tmp/nms';
gtDir='/media/fan/dv2/backup/fan_part3/home/fy/fan/hed/data/crack/testcrop';
edgesEvalDir_crack('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
figure; edgesEvalPlot(resDir,model);
close all