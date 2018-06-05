X=Ndata(:,1:20)'; %Features
Group=Ndata(:,21)'; %Output

%% For Ranking by T-tests

[rank_t_test, ttest_value] = rankfeatures(Ndata(:,1:20)',Ndata(:,21)');

%% For Kullback-Leibler distance (Relative Entropy) 

[rank_entropy, entropy_value] = rankfeatures(X,Group,'Criterion','entropy');

%% Ranking by Bhattacharya test
[rank_bhatta, bhatta_value] = rankfeatures(X,Group,'Criterion','bhattacharyya');

%%
subplot(1,3,1);
bar(ttest_value);
xlabel('Features');
ylabel('T-tests');

subplot(1,3,2);
bar(bhatta_value);
xlabel('Features');
ylabel('Bhattacharya distance ');

subplot(1,3,3);
bar(entropy_value);
xlabel('Features');
ylabel('Kullback-Leibler distance or Relative Entropy ');



%%
A = [ 20.19890589	0.061846697	0.559412363
30.81814159	0.165124094	2.825853289
16.65637772	0.043520066	0.403914256
33.57100655	0.168373966	2.130177848
3.772844472	0.002336227	0.021008408
44.45284416	0.293220801	5.440928094
2.060039461	0.016576362	0.566174884
4.922008996	0.020149808	0.611680596
31.73407953	0.15734055	2.12382843
21.60311224	0.071015427	0.671579556
9.86802971	0.018383641	0.232038609
20.19890589	0.061846697	0.559412363
84.74335178	0.627711007	9.1180101
7.788809217	0.016284436	0.307346357
9.48070915	0.030742127	0.722278898
10.99063402	0.022994724	0.298128263
11.18610533	0.02926233	0.50623363
11.27037462	0.023714105	0.298331979
11.06500415	0.023025917	0.292660378
1.783606726	0.00276668	0.077556522]; %Output from all three tests

B = A';
