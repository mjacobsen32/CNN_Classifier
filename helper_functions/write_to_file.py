from constants import output_file_name


def write_to_file(information):
    with open(output_file_name, 'w') as f:
        f.write(information)
