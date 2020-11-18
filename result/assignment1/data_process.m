for i=0:1:9
    binary_data = csvread("trail-"+i+"-binary.txt");
    hold on
    figure(1)
    plot(binary_data(:,1), binary_data(:,2));
end

xlabel("Epoch Number", "FontSize", 14);
ylabel("Total Error", "FontSize", 14);
title("10 trials of learning process using binary representation", "FontSize", 18);

for i=0:1:9
    bipolar_data = csvread("trail-"+i+"-bipolar.txt");
    hold on
    figure(2)
    plot(bipolar_data(:,1), bipolar_data(:,2));
end

xlabel("Epoch Number", "FontSize", 14);
ylabel("Total Error", "FontSize", 14);
title("10 trials of learning process using bipolar representation", "FontSize", 18);