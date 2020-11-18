%% 2.a
off_06 = csvread("result-0.6-offPolicy.txt");

figure();
plot(off_06(:,1), off_06(:,3));
xlabel("rounds");
ylabel("win rate of last 100 rounds");
title("Q-Learning");

%% 2.b
on_06 = csvread("result-0.6-onPolicy.txt");
figure();

hold on;
plot(off_06(:,1), off_06(:,3));
plot(on_06(:,1), on_06(:,3));
legend("offPolicy", "onPolicy");
hold off;

xlabel("rounds");
ylabel("win rate of last 100 rounds");
title("Q-Learning of Different Policies");

%% 2.c
terminal = csvread("result-0.6-offPolicy-terminal.txt");

figure();

hold on;
plot(off_06(:,1), off_06(:,3));
plot(terminal(:,1), terminal(:,3));
legend("Intermidiate Reward", "Terminal Reward");
hold off;

xlabel("rounds");
ylabel("win rate of last 100 rounds");
title("Q-Learning with Different Reward");

%% 3.a
off_04 = csvread("result-0.4-offPolicy.txt");
off_02 = csvread("result-0.2-offPolicy.txt");
off_00 = csvread("result-0.0-offPolicy.txt");

figure();

hold on;
plot(off_06(:,1), off_06(:,3));
plot(off_04(:,1), off_04(:,3));
plot(off_02(:,1), off_02(:,3));
plot(off_00(:,1), off_00(:,3));
legend("0.6", "0.4", "0.2", "0");
hold off;

xlabel("rounds");
ylabel("win rate of last 100 rounds");
title("Q-Learning with Different Exploration Rate");
