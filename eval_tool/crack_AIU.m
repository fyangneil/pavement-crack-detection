
resultsPath='/data/crack/testresults/hed_fuse_fpn_ada_python_v1_iter_12000';
gruImgPath='/data/crack/testcrop';
get_pIUPR(resultsPath,gruImgPath);
get_AIU(resultsPath);
function get_pIUPR(resultsPath,gruImgPath)
%% compute pIU (IU for positives), precion, and recall on each image over diifferent threshold
class_num = 4;
if ~(exist([resultsPath,'-evalIu'],'dir')==7)
    mkdir([resultsPath,'-evalIu']);
end
outputpath=[resultsPath,'-evalIu'];
predImgList=dir([resultsPath,'/*.mat']);
gruImgList=dir([gruImgPath,'/*.png']);
sourceImgList=dir([gruImgPath,'*.jpg']);
mPIU=0;%mean Positive IU
mPre=0;%mean precision
mRc=0;% mean recall
thrs=linspace(0.01,0.99,99)';
% num denotes number of testing images
parfor i = 1:length(gruImgList)
fid=fopen(fullfile(outputpath,[gruImgList(i).name(1:end-3),'txt']),'w');
for ind=1:length(thrs)
    threshold=thrs(ind);
    confusion = zeros(class_num) ;
    % Load gt label and prediction
    lb = imread(fullfile(gruImgPath,gruImgList(i).name));   % replace with image names
    lb(lb>0)=1;
    lb=lb+1;
    pred=load(fullfile(resultsPath,predImgList(i).name));
    pred=pred.predmap;
    if length(size(pred))==3
        % if pred is 3 channel, convert to gray image
        pred = rgb2gray(pred);
    end
    pred(pred>=max(threshold,eps))=1;
    pred(pred<max(threshold,eps))=0;
    pred=pred+1;
    % Accumulate errors, 0 is ignored
    ok = lb > 0;
    confusion = confusion + accumarray([lb(ok),pred(ok)],1,[class_num class_num]) ;

    [pIU,precision,recall] = getMetrics(confusion);
fprintf(fid,'%10g %10g %10g %10g\n',[threshold pIU precision recall]');
end
fclose(fid);
end
end
function get_AIU(resultsDir)
%% compute mean pIU, precision, recall over dataset
suffix='-evalIu';
evalImgPath=[resultsDir,suffix];
ImgList=dir([evalImgPath,'/*.txt']);
mTpIUPR=0;% mean evaluation metrics over dataset
for imgInd=1:length(ImgList)
    % load evaluation metrics pIU, Precision, Recall over each Threshold (TpIUPR) for each image
    TpIUPR=importdata(fullfile(evalImgPath,[ImgList(imgInd).name(1:end-3),'txt']));
    if size(TpIUPR,2)==4
    mTpIUPR=mTpIUPR+TpIUPR;
    end
end
mTpIUPR=mTpIUPR/length(ImgList);
fid=fopen(fullfile(evalImgPath,['mTpIUPR','.txt']),'w');
fprintf(fid,'%10g %10g %10g %10g\n',mTpIUPR');
fclose(fid);
fprintf('AIU %.4f\n',mean(mTpIUPR(:,2)));
end
% -------------------------------------------------------------------------
function [pIU,precision,recall] = getMetrics(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;%positives
res = sum(confusion,1)' ;%results
tp = diag(confusion);
fp=res - tp;
precision=tp./max(eps,(tp+fp));
precision=precision(2);
recall=tp./pos;
recall=recall(2);
IU = tp ./ max(1, pos + res - tp);
pIU=IU(2);%IU for positives
end
