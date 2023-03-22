import os

def get_files_list(path):
    """获取目录下的所有文件"""
    all_files = [f for f in os.listdir(path)]
    return all_files

def getcwd():
    return os.getcwd()

def mkdir(path):
    """
    如果不存在则创建目录
    存在则打印信息
    """
    # 去除首位空格，去除尾部 \ 符号
    path=path.strip().rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path) # 创建目录操作函数
    else:
        print(path+' 目录已存在')

def get_file_of_prefix(file_prefix):
    """
    根据文件前面的前缀来匹配文件（如有相同的则返回第一个）
    基于当前目录
    没有返回None
    """
    if "/" in file_prefix:
        file_split=file_prefix.split("/")
        file_split=[i for i in file_split if i]
        file_dir="/".join(file_split[:-1])
        file_prefix=file_split[-1]
        files = get_files_list(file_dir)
    else:
        files = get_files_list(os.getcwd())

    for f in files:
        if f[:len(file_prefix)]==file_prefix:
            return file_dir+"/"+f



