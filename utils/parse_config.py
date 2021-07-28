def parse_model_cfg(path):
    """
    Purpose:
        parse the file to get the layer info for building the model
    Args:
        path : str, cfg file
    Return:
        list(dictionary)
    Process:
        1. read file
        2. parse layers:
            split into lines
            get rid of empty lines
            ignore comment lines, start with '#'
        3. add to list

    """

    file = open(path, 'r')
    file_lines = file.read().split('\n') # split into lines
    file_lines = [l for l in file_lines if len(l) > 0] # get rid of empty lines

    model_layers = [] # to store layer dict
    layer = {} # dictionary to store layer info

    write_flag = False # whether it is the end of layer
    for l in file_lines:
        if l[0] == '[':
            if write_flag:
                model_layers.append(layer)
                layer = {}
            layer['type'] = l[1:-1]
            write_flag = True

        elif l[0] == '#':
            continue
        else:
            key,value = l.strip().split('=')
            layer[key] = value

    model_layers.append(layer)

    return model_layer



