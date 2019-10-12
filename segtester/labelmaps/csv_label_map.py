import csv
import numpy as np


class CSVLabelMap:
    def __init__(self, config):
        self.csv_path = config.csv_path

    def get_label_text(self, id_col_name, text_col_name, default_value='object', default_key=0):
        max_from_label = 0
        label_map = {default_key: default_value}
        with open(self.csv_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                key = int(row[id_col_name]) if row[id_col_name] != '' else default_key
                val = row[text_col_name]
                max_from_label = max(max_from_label, key)
                if key not in label_map:  # take the first one matching
                    label_map[key] = val

        labels = np.array(list(label_map.values()))
        np_label_map = np.repeat(np.array(default_value, dtype=labels.dtype), max_from_label+1)
        np_label_map[list(label_map.keys())] = labels
        return np_label_map

    def get_inverse_text_map(self, id_col_name, text_col_name, default_value=0, default_key='object'):
        label_map = {default_key: default_value}
        with open(self.csv_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                val = int(row[id_col_name]) if row[id_col_name] != '' else default_key
                key = row[text_col_name]
                if key not in label_map:  # take the first one matching
                    label_map[key] = val
        return label_map

    def get_unique_values(self, id_col_name, default_key=0):
        keys = {0, }
        with open(self.csv_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                key = int(row[id_col_name]) if row[id_col_name] != '' else default_key
                keys.add(key)
        return list(keys)

    def get_label_map(self, from_col, to_col, default_value=0, default_key=0):
        string_label_map = self.get_label_text(from_col, to_col,
                                               default_value=str(default_value),
                                               default_key=default_key)
        string_label_map[string_label_map == ''] = 0

        return string_label_map.astype(np.int)
