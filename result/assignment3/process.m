%%
% NN-30
NN_30_00_02 = csvread("NN-30/0.0-0.2-trial-0.txt");
NN_30_00_01 = csvread("NN-30/0.0-0.1-trial-0.txt");
NN_30_00_005 = csvread("NN-30/0.0-0.05-trial-0.txt");
NN_30_00_001 = csvread("NN-30/0.0-0.01-trial-0.txt");
NN_30_02_001 = csvread("NN-30/0.2-0.01-trial-0.txt");
NN_30_04_001 = csvread("NN-30/0.4-0.01-trial-0.txt");
NN_30_06_001 = csvread("NN-30/0.6-0.01-trial-0.txt");
NN_30_08_001 = csvread("NN-30/0.8-0.01-trial-0.txt");
NN_30_1_001 = csvread("NN-30/1.0-0.01-trial-0.txt");

% hidden neurons
NN_40_00_001 = csvread("NN-40/0.0-0.01-trial-0.txt");
NN_60_00_001 = csvread("NN-60/0.0-0.01-trial-0.txt");
NN_80_00_001 = csvread("NN-80/0.0-0.01-trial-0.txt");

% NN-80
NN_80_00_005 = csvread("NN-80/0.0-0.05-trial-0.txt");
NN_80_00_01 = csvread("NN-80/0.0-0.1-trial-0.txt");
NN_80_00_02 = csvread("NN-80/0.0-0.2-trial-0.txt");
NN_80_02_001 = csvread("NN-80/0.2-0.01-trial-0.txt");
NN_80_04_001 = csvread("NN-80/0.4-0.01-trial-0.txt");
NN_80_06_001 = csvread("NN-80/0.6-0.01-trial-0.txt");
NN_80_08_001 = csvread("NN-80/0.8-0.01-trial-0.txt");
%%
% Figure 1
figure();
hold on;
plot(NN_30_00_02(:,1), NN_30_00_02(:,2));
plot(NN_30_00_01(:,1), NN_30_00_01(:,2));
plot(NN_30_00_005(:,1), NN_30_00_005(:,2));
plot(NN_30_00_001(:,1), NN_30_00_001(:,2));
legend("learning rate: 0.2", "learning rate: 0.1", "learning rate: 0.05", "learning rate: 0.01");
title("Learning with 30 hidden neurons, 0.0 momentum in 10000 epoches");
ylabel("Error");
xlabel("# Epochs");

%%
% Figure 2
figure();
hold on;
plot(NN_30_00_001(:, 1), NN_30_00_001(:, 2));
plot(NN_30_02_001(:, 1), NN_30_02_001(:, 2));
plot(NN_30_04_001(:, 1), NN_30_04_001(:, 2));
plot(NN_30_06_001(:, 1), NN_30_06_001(:, 2));
plot(NN_30_08_001(:, 1), NN_30_08_001(:, 2));
plot(NN_30_1_001(:, 1), NN_30_1_001(:, 2));
legend("momentum: 0.0", "momentum: 0.2", " momentum: 0.4", "momentum: 0.6", "momentum: 0.8", "momentum: 1.0");
title("Learning with 30 hidden neurons, 0.01 learning rate in 10000 epoches");
ylabel("Error");
xlabel("# Epochs");

%%
% Figure 3
figure();
hold on;
plot(NN_30_00_001(:,1), NN_30_00_001(:,2));
plot(NN_40_00_001(:,1), NN_40_00_001(:,2));
plot(NN_60_00_001(:,1), NN_60_00_001(:,2));
plot(NN_80_00_001(:,1), NN_80_00_001(:,2));
legend("hidden neurons: 30", "hidden neurons: 40", "hidden neurons: 60", "hidden neurons: 80");
title("Learning with 0.01 learning rate and 0.0 momentum in 10000 epoches");
ylabel("Error");
xlabel("# Epoches");

%%
% Figure 4
figure();
hold on;
plot(NN_80_00_001(:, 1), NN_80_00_001(:, 2));
plot(NN_80_00_005(:, 1), NN_80_00_005(:, 2));
plot(NN_80_00_01(:, 1), NN_80_00_01(:, 2));
plot(NN_80_00_02(:, 1), NN_80_00_02(:, 2));
legend("learning rate: 0.01", "learning rate: 0.05", "learning rate: 0.1", "learning rate: 0.2");
title("Learning with 80 hidden neurons and 0.0 momentum in 10000 epoches");
ylabel("Error");
xlabel("# Epoches");
%%
% Figure 5
figure();
hold on;
plot(NN_80_00_001(:, 1), NN_80_00_001(:, 2));
plot(NN_80_02_001(:, 1), NN_80_02_001(:, 2));
plot(NN_80_04_001(:, 1), NN_80_04_001(:, 2));
plot(NN_80_06_001(:, 1), NN_80_06_001(:, 2));
plot(NN_80_08_001(:, 1), NN_80_08_001(:, 2));
legend("momentum: 0.0", "momentum: 0.2", "momentum: 0.4", "momentum: 0.6", "momentum: 0.8");
title("Learning with 80 hidden neurons and 0.01 learning rate in 10000 epoches");
ylabel("Error");
xlabel("# Epoches");

%%
% Figure 6
figure();
plot(NN_80_06_001(:, 1), NN_80_06_001(:, 2));
title("Learning with 80 hidden neurons and 0.01 learning rate and 0.6 momentum in 10000 epoches");
ylabel("Error");
xlabel("# Epoches");

%%
% Question 5
NN_05_00 = csvread("feature-factor/NN-0.5-0.0-offPolicy.txt");
NN_05_00_reward = csvread("feature-factor/NN-0.5-0.0-offPolicy-rewards.txt");
NN_05_02 = csvread("feature-factor/NN-0.5-0.2-offPolicy.txt");
NN_05_02_reward = csvread("feature-factor/NN-0.5-0.2-offPolicy-rewards.txt");
NN_05_04 = csvread("feature-factor/NN-0.5-0.4-offPolicy.txt");
NN_05_04_reward = csvread("feature-factor/NN-0.5-0.4-offPolicy-rewards.txt");
NN_05_06 = csvread("feature-factor/NN-0.5-0.6-offPolicy.txt");
NN_05_06_reward = csvread("feature-factor/NN-0.5-0.6-offPolicy-rewards.txt");
NN_05_08 = csvread("feature-factor/NN-0.5-0.8-offPolicy.txt");
NN_05_08_reward = csvread("feature-factor/NN-0.5-0.8-offPolicy-rewards.txt");
NN_05_10 = csvread("feature-factor/NN-0.5-1.0-offPolicy.txt");
NN_05_10_reward = csvread("feature-factor/NN-0.5-1.0-offPolicy-rewards.txt");

%% 
% Figure 7
figure();
plot(NN_05_08(:, 1), NN_05_08(:, 3));
title("Learning process of win rate with future factor of 0.8");
ylabel("Win rate of the last 100 rounds");
xlabel("Rounds");

%%
% Figure 8
figure();
plot(NN_05_08_reward(:, 1));
title("Learning process of cumulative reward with future factor of 0.8");
ylabel("Cumulative reward of the last 200 rewards");
xlabel("State");

%%
% Figure 9
figure();
hold on;
plot(NN_05_00(:, 1), NN_05_00(:, 3));
plot(NN_05_02(:, 1), NN_05_02(:, 3));
plot(NN_05_04(:, 1), NN_05_04(:, 3));
plot(NN_05_06(:, 1), NN_05_06(:, 3));
plot(NN_05_08(:, 1), NN_05_08(:, 3));
plot(NN_05_10(:, 1), NN_05_10(:, 3));
title("Learning process of win rates with different future factor gamma");
legend("gamma: 0.0", "gamma: 0.2", "gamma: 0.4", "gamma: 0.6", "gamma: 0.8", "gamma: 1.0");
ylabel("Win rate of the last 100 rounds");
xlabel("Rounds");

%%
% Figure 10
figure();
hold on;
plot(NN_05_00_reward(:, 1));
plot(NN_05_02_reward(:, 1));
plot(NN_05_04_reward(:, 1));
plot(NN_05_06_reward(:, 1));
plot(NN_05_08_reward(:, 1));
plot(NN_05_10_reward(:, 1));
title("Learning process of cumulative reward with different future factor gamma");
legend("gamma: 0.0", "gamma: 0.2", "gamma: 0.4", "gamma: 0.6", "gamma: 0.8", "gamma: 1.0");
ylabel("Cumulative reward of the last 200 rewards");
xlabel("State");

%%
% Last N input
NN_last_0 = csvread("lastN/NN-0.5-0.8-0-offPolicy.txt");
NN_last_1 = csvread("lastN/NN-0.5-0.8-1-offPolicy.txt");
NN_last_2 = csvread("lastN/NN-0.5-0.8-2-offPolicy.txt");
NN_last_4 = csvread("lastN/NN-0.5-0.8-4-offPolicy.txt");
NN_last_8 = csvread("lastN/NN-0.5-0.8-8-offPolicy.txt");

%%
% Figure 11
figure();
hold on;
plot(NN_last_1(:, 1), NN_last_1(:, 3));
plot(NN_last_2(:, 1), NN_last_2(:, 3));
plot(NN_last_4(:, 1), NN_last_4(:, 3));
plot(NN_last_8(:, 1), NN_last_8(:, 3));
title("Win rate using N last experience");
legend("N: 1", "N: 2", "N: 4", "N: 8");
ylabel("Win rate of the last 100 rounds");
xlabel("Rounds");

