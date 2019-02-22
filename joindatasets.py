import sys, os
from transform_r_data_to_python_format import *

def join_datasets_python_format(ts_in_files_1,file_to_join_1,file_to_join_2, name_final_file):

    train_file_to_join_1 = file_to_join_1 + "_TRAIN"
    test_file_to_join_1 = file_to_join_1 + "_TEST"
    file_1 = []
    file_1.append(train_file_to_join_1)
    file_1.append(test_file_to_join_1)

    train_file_to_join_2 = file_to_join_2 + "_TRAIN"
    test_file_to_join_2 = file_to_join_2 + "_TEST"
    file_2 = []
    file_2.append(train_file_to_join_2)
    file_2.append(test_file_to_join_2)

    files_to_join = []
    files_to_join.append(file_1)
    files_to_join.append(file_2)

    train_name_final_file = name_final_file + "_TRAIN"
    test_name_final_file = name_final_file + "_TEST"
    final_file = []
    final_file.append(train_name_final_file)
    final_file.append(test_name_final_file)

    for i in range (0,2):
        ts_count = ts_in_files_1 + 1
        ts_index_value = 1
        with open(final_file[i], "w" ) as file_to_write:
            for j in range (0,2):
                with open(files_to_join[j][i], "r" ) as file_to_read:
                    for line in file_to_read:
                        if j == 0:
                            #first file case
                            file_to_write.write(line)
                        else:
                            #second file case
                            list_line = line.split()
                            if  ts_index_value != int(list_line[0]):
                                ts_count = ts_count + 1
                                ts_index_value = int(list_line[0])

                            list_line[0] = str(int(ts_count))

                            file_to_write.write(" ".join(list_line) + "\n")

#main
variables_dimension = 6
ts_number = 500
ts_length = 100

file_1_name = 'var_6_cov_diagonal.csv'
file_1_to_write_name = 'DATASETS/SYNTETICS_R/var_6_cov_diag'
class_name_to_use = 0
transform_r_file_to_python_format_function(file_1_name, file_1_to_write_name,variables_dimension,ts_number,ts_length,class_name_to_use)

file_2_name = 'var_6_cov_toep.csv'
file_2_to_write_name = 'DATASETS/SYNTETICS_R/var_6_covar_toep'
class_name_to_use = 1
transform_r_file_to_python_format_function(file_2_name, file_2_to_write_name,variables_dimension,ts_number,ts_length,class_name_to_use)

name_final_file = 'DATASETS/SYNTETICS_R/var_6_cov_diag_toep'

join_datasets_python_format(ts_number/2,file_1_to_write_name,file_2_to_write_name, name_final_file)
