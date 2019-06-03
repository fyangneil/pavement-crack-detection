model='FPHB';
fprintf('%s\n',model);
resDir='/data/crack/testresults/tmp/nms';
gtDir='/data/crack/testcrop';
edgesEvalDir_crack('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
figure; edgesEvalPlot(resDir,model);
close all
