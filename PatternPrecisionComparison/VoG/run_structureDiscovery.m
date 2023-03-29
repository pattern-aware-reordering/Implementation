ijk_list = csvread("../probabilities.csv", 1, 0);
for count=1:length(ijk_list)
    ijk = ijk_list(count, :);
    i = ijk(1);
    j = ijk(2);
    k = ijk(3);
    input_file = string(compose('../data/synthetic-%d-%d-%d.out', i, j, k));
    % input_file = 'DATA/cliqueStarClique.out';
    unweighted_graph = input_file;
    output_model_greedy = 'DATA';
    output_model_top10 = 'DATA';

    addpath('STRUCTURE_DISCOVERY');

    orig = spconvert(load(input_file));
    orig(max(size(orig)),max(size(orig))) = 0;
    orig_sym = orig + orig';
    [i,j,k] = find(orig_sym);
    orig_sym(i(find(k==2)),j(find(k==2))) = 1;
    orig_sym_nodiag = orig_sym - diag(diag(orig_sym));

    disp('==== Running VoG for structure discovery ====')
    global model; 
    model = struct('code', {}, 'edges', {}, 'nodes1', {}, 'nodes2', {}, 'benefit', {}, 'benefit_notEnc', {});
    global model_idx;
    model_idx = 0;
    SlashBurnEncode( orig_sym_nodiag, 2, output_model_greedy, false, false, 3, unweighted_graph);
    % quit
end