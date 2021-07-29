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

    file = open(path,'r')
    lines = file.read().split('\n') 
    lines = [x for x in lines if len(x) >0] # 去除空行
    lines = [x for x in lines if x[0] != '#'] #去除注释行
    lines = [x.strip() for x in lines] 

    block = {} # store as dictionary
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block) # 把上一个block添加到blocks中
                block= {}
            block['type'] = line[1:-1].strip() # 下一个block的类型名称
        else:
            key,value = line.split('=')
            block[key.strip()] = value.strip()

    blocks.append(block) # 把最后一个block 也加入

    return blocks



