clear; clc;
path_to_input='/data/crack/testresults/hed_fuse_fpn_ada_python_v1_iter_12000';
path_to_output='/data/crack/testresults/nms';
mkdir(path_to_output);
iids = dir(fullfile(path_to_input, '*.mat'));
for i = 1:length(iids)
    %edge = imread(fullfile(path_to_input, iids(i).name));\
    load(fullfile(path_to_input, iids(i).name));
    predmap=predmap;
    [Ox, Oy] = gradient2(convTri(predmap, 4));
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
    predmap = edgesNmsMex(predmap, O, 2, 5, 1.01, 8);
    imwrite(predmap, fullfile(path_to_output, [iids(i).name(1:end-4) '.png']));
    save(fullfile(path_to_output, [iids(i).name(1:end-4) '.mat']),'predmap');
end
