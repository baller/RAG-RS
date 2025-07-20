def subset_dict_by_filename(files_to_subset, dictionary):
    return {file : dictionary[file] for file in files_to_subset}

def filter_labels_by_threshold(labels_dict, area_threshold = 0.07):
    """
    Parameters
    ----------
    labels_dict: dict, {filename1: [(label, area)],
                        filename2: [(label, area), (label, area)],
                        ...
                        filenameN: [(label, area), (label, area)]}
    area_threshold: float
    
    Returns
    -------
    filtered: dict, {filename1: [label],
                     filename2: [label, label],
                     ...
                     filenameN: [label, label]}
    """
    filtered = {}
    
    for img in labels_dict:
        for lbl, area in labels_dict[img]:
            # if area greater than threshold we keep the label
            if area > area_threshold:
                # init the list of labels for the image
                if img not in filtered:
                    filtered[img] = []
                # add only the label, since we won't use area information further
                filtered[img].append(lbl)
                
    return filtered


def get_treesat_classes(data_path, verbose=True):
    """
    动态从TreeSAT标签文件中读取所有类别
    
    Parameters
    ----------
    data_path: str
        TreeSAT数据集路径
    verbose: bool
        是否打印类别信息
    
    Returns
    -------
    classes: list
        所有树种类别的列表
    """
    import json
    import os
    
    labels_file = os.path.join(data_path, "labels", "TreeSatBA_v9_60m_multi_labels.json")
    
    try:
        with open(labels_file, 'r') as f:
            label_data = json.load(f)
        
        # 提取所有唯一的类别
        all_classes = set()
        for filename, labels in label_data.items():
            for label, area in labels:
                all_classes.add(label)
        
        classes = sorted(list(all_classes))  # 排序以确保一致性
        
        if verbose:
            print(f"从数据集中发现 {len(classes)} 个树种类别:")
            for i, cls in enumerate(classes):
                print(f"  {i+1:2d}. {cls}")
        
        return classes
        
    except FileNotFoundError:
        if verbose:
            print(f"警告: 标签文件 {labels_file} 不存在，使用默认类别列表")
        # 包含Pseudotsuga的完整默认类别列表
        classes = [
            'Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus',
            'Corylus', 'Fagus', 'Fraxinus', 'Larix', 'Picea', 
            'Pinus', 'Populus', 'Pseudotsuga', 'Quercus', 'Salix', 'Tilia'
        ]
        if verbose:
            print(f"使用默认的 {len(classes)} 个类别")
        return classes
    
    except Exception as e:
        if verbose:
            print(f"读取标签文件时出错: {e}")
            print("使用扩展的默认类别列表")
        classes = [
            'Abies', 'Acer', 'Alnus', 'Betula', 'Carpinus',
            'Corylus', 'Fagus', 'Fraxinus', 'Larix', 'Picea', 
            'Pinus', 'Populus', 'Pseudotsuga', 'Quercus', 'Salix', 'Tilia'
        ]
        return classes