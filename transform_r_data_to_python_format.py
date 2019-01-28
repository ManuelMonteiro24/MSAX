import csv

#'variables_2_covar_identity.csv'
file_name = 'variables_2_covar_identity.csv'
file_to_write_name = 'DATASETS/SYNTETICS_R/variables_2_covar_identity'
variables_dimension = 2
ts_number = 100
ts_length = 100
class_name_to_use = 0
test_flag = 0

#####

series_index = 1
series_time_index = 1

train_file_name = file_to_write_name + "_TRAIN"
test_file_name = file_to_write_name + "_TEST"
#open file to write
with open(train_file_name, "w" ) as train_file:
    with open(test_file_name, "w" ) as test_file:

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    #first line case
                    line_count += 1
                else:
                    output_str = str(series_index) + " " + str(series_time_index) + " " + str(class_name_to_use)
                    for j in range(1,variables_dimension+1):
                        output_str += " " + str(row[j])

                    #write phase
                    if test_flag ==1:
                        test_file.write(output_str + "\n")
                    else:
                        train_file.write(output_str + "\n")

                    line_count += 1
                    series_time_index = series_time_index + 1
                    if line_count > ts_length:
                        line_count = 1
                        if series_index == ts_number/2:
                            test_flag=1
                            series_index = 1
                        else:
                            series_index = series_index + 1
                        series_time_index = 1
