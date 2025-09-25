from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse
import pickle
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, accuracy_score
import os
import joblib
import json
import random

random.seed(42)
torch.manual_seed(42)

def replace_nan_in_list(data_list):
    for i in range(len(data_list)):
        data_list[i] = np.nan_to_num(data_list[i], nan=0.0, posinf=1e4, neginf=-1e4)
        if np.isnan(data_list[i]).any() or np.isinf(data_list[i]).any():
            print("aa")
    return data_list

def filter_data_by_gpt_labels(train_gpt_labels, test_gpt_labels, 
                              train_labels, train_data, train_position, train_topk_position,
                              test_labels, test_data, test_position, test_topk_position, args):
    """
    Filter all data based on the 'label' field in GPT labels.
    Only keep data where label=True.
    
    Args:
        train_gpt_labels: List of lists containing train GPT label data
        test_gpt_labels: List of lists containing test GPT label data
        train_labels: Training labels array
        train_data: Training data
        train_position: Training position data
        train_topk_position: Training top-k position data
        test_labels: Test labels array
        test_data: Test data
        test_position: Test position data
        test_topk_position: Test top-k position data
    
    Returns:
        Tuple of filtered data in the same order as input
    """
    
    # Get train indices where label=True
    train_keep_indices = []
    for i, label_group in enumerate(train_gpt_labels):
        # Each group is a list with one dictionary
        if train_labels[i] == "Induction":
            continue
        if args.all_train:
            train_keep_indices.append(i)
        elif len(label_group) > 0 and label_group[0].get('label', False):
            train_keep_indices.append(i)
        
    
    # Get test indices where label=True
    test_keep_indices = []
    for i, label_group in enumerate(test_gpt_labels):
        # Each group is a list with one dictionary
        if test_labels[i] == "Induction":
            continue
        if args.all_train:
            test_keep_indices.append(i)
        if len(label_group) > 0 and label_group[0].get('label', False):
            test_keep_indices.append(i)

    train_data = replace_nan_in_list(train_data)
    test_data = replace_nan_in_list(test_data)
    # Filter train data
    filtered_train_labels = train_labels[train_keep_indices] if hasattr(train_labels, '__getitem__') else [train_labels[i] for i in train_keep_indices]
    filtered_train_data = [train_data[i] for i in train_keep_indices] if isinstance(train_data, list) else train_data[train_keep_indices]
    filtered_train_position = [train_position[i] for i in train_keep_indices] if isinstance(train_position, list) else train_position[train_keep_indices]
    filtered_train_topk_position = [train_topk_position[i] for i in train_keep_indices] if isinstance(train_topk_position, list) else train_topk_position[train_keep_indices]
    
    # Filter test data
    filtered_test_labels = test_labels[test_keep_indices] if hasattr(test_labels, '__getitem__') else [test_labels[i] for i in test_keep_indices]
    filtered_test_labels = ["Semantic Understanding" if w == "semantic Understanding" else w for w in filtered_test_labels]
    filtered_test_data = [test_data[i] for i in test_keep_indices] if isinstance(test_data, list) else test_data[test_keep_indices]
    filtered_test_position = [test_position[i] for i in test_keep_indices] if isinstance(test_position, list) else test_position[test_keep_indices]
    filtered_test_topk_position = [test_topk_position[i] for i in test_keep_indices] if isinstance(test_topk_position, list) else test_topk_position[test_keep_indices]
    
    print(f"Original train data size: {len(train_gpt_labels)}")
    print(f"Filtered train data size: {len(train_keep_indices)}")
    print(f"Original test data size: {len(test_gpt_labels)}")
    print(f"Filtered test data size: {len(test_keep_indices)}")
    
    
    return (filtered_train_data, filtered_test_data, filtered_train_labels, filtered_test_labels, filtered_train_position, filtered_test_position, filtered_train_topk_position, filtered_test_topk_position)


def train_probes_multi(args, all_X_train, y_train):
    
    probes = []
    for layer in tqdm(range(args.layer_num), desc="train_probes"): 
        for head in range(args.heads_num): 
            functions_acc = []
            X_train = all_X_train[:,layer,head,:]
            # X_test = all_X_test[:,layer,head,:]
    
            clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)

            probes.append(clf)

    return probes

def train_probes_single(args, all_X_train, y_train):
    probes = []
    model = LogisticRegression(random_state=42, max_iter=1000)

    n_index, n_layer, n_head, n_features = all_X_train.shape
    recombined_X_train = all_X_train.reshape(-1, n_features)
    
    # 重复y_train: 每个index重复 layer*head 次
    recombined_y_train = np.repeat(y_train, n_layer * n_head)

    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(recombined_X_train))
    recombined_X_train = recombined_X_train[shuffle_indices]
    recombined_y_train = recombined_y_train[shuffle_indices]
    
    print(recombined_X_train.shape)
    print(recombined_y_train.shape)
    clf = model.fit(recombined_X_train, recombined_y_train)
    probes.append(clf)
    # with open(f'./trained_classifiers/logistic_regression/{args.model}_multi.joblib', 'wb') as f:
    #     joblib.dump(probes, f)

    return probes


def load_data(model_name, args):

    if "7B" in model_name or "8B" in model_name or "4b" in model_name:
        with open(f'./head_results/{model_name}_train_1000_head_wise_train_0.pkl', 'rb') as f:
            train_data_0 = pickle.load(f)
        print("load 0...")
        with open(f'./head_results/{model_name}_train_1000_head_wise_train_1.pkl', 'rb') as f:
            train_data_1 = pickle.load(f)
        print("load 1...")
        with open(f'./head_results/{model_name}_train_1000_head_wise_train_2.pkl', 'rb') as f:
            train_data_2 = pickle.load(f)
        print("load 2...")
        with open(f'./head_results/{model_name}_train_1000_head_wise_train_3.pkl', 'rb') as f:
            train_data_3 = pickle.load(f)
        print("load 3...")
        train_data = train_data_0 + train_data_1 + train_data_2 + train_data_3
    else:
        with open(f'./head_results/{model_name}_train_1000_head_wise_train.pkl', 'rb') as f:
            train_data = pickle.load(f)
    with open(f'./head_results/output_{model_name}_train_1000_train_with_gpt_label.json', "rb") as f:
        train_gpt_labels = json.load(f)
    with open(f'./head_results/output_{model_name}_test_1000_test_with_gpt_label.json', "rb") as f:
        test_gpt_labels = json.load(f)
    with open(f'./head_results/train_labels.npy', "rb") as f:
        train_labels = np.load(f)
    with open(f'./head_results/test_labels.npy', "rb") as f:
        test_labels = np.load(f)
    with open(f'./head_results/{model_name}_test_1000_head_wise_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(f"./head_results/{model_name}_train_1000_token_positions_train.pkl", 'rb') as f:
        train_position = pickle.load(f)
    with open(f"./head_results/{model_name}_test_1000_token_positions_test.pkl", 'rb') as f:
        test_position = pickle.load(f)
    with open(f'./head_results/{model_name}_train_1000_topk_position_train.pkl', "rb") as f:
        train_topk_position = pickle.load(f)
    with open(f'./head_results/{model_name}_test_1000_topk_position_test.pkl', "rb") as f:
        test_topk_position = pickle.load(f)
    #print(len(train_data))
    return filter_data_by_gpt_labels(train_gpt_labels, test_gpt_labels, train_labels, train_data, train_position, train_topk_position, test_labels, test_data, test_position, test_topk_position, args)

def get_head_weight(train, test, layer_num, heads_num, dim, position, head_num, layer_bias,
                    train_position, test_position, train_topk_position, test_topk_position):
    def extract_features(data, position_data, topk_data, is_train=True):
        features = []
        for i in range(len(data)):
            token_num = data[i].shape[1]
            #print("token_num", token_num)
            #print("topk_data", len(topk_data))
            token_select = 0
            if position == "first":
                token_select = [0]
            elif position == "last":
                token_select = [token_num - 1]
            elif position == "meaning":
                token_select = [position_data[i][0]]
            elif position == "topk":
                token_select = [x for x in topk_data[i] if 0 <= x < token_num]
            elif position == "full":
                token_select = [x for x in position_data[i] if 0 <= x < token_num]

            if not token_select:
                token_select = [x for x in position_data[i] if 0 <= x < token_num]

            reshaped = data[i].reshape(layer_num, token_num, heads_num, dim)
            reshaped = reshaped[:, token_select, :, :]

            #if position in ["topk", "full"]:
            reshaped = np.mean(reshaped, axis=1)

            if layer_bias:
                layer_means = reshaped.mean(axis=1)
                expanded = np.expand_dims(layer_means, axis=1).repeat(heads_num, axis=1)
                reshaped = np.concatenate([reshaped, expanded], axis=-1)
                reshaped = reshaped.reshape(layer_num * heads_num, dim * 2)
            else:
                reshaped = reshaped.reshape(layer_num * heads_num, dim)

            features.append(reshaped if head_num == -1 else reshaped[head_num])
        return np.array(features)

    train_feat = extract_features(train, train_position, train_topk_position)
    test_feat = extract_features(test, test_position, test_topk_position, is_train=False)
    return train_feat, test_feat

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def per_classifier_label_f1_single(args, classifiers, target_label, train_data, all_X_test, labels,score ='f1'):
    scores = []
    for layer in range(args.layer_num): 
        for head in range(args.heads_num): 
            #num_layer, num_head = flattened_idx_to_layer_head(idx, args.heads_num)
            # _,X_test =  get_head_weight(train_data,test_data, head_num = idx)
            X_test = all_X_test[:, layer, head, :]  # Select the features for the current classifier
            y_pred = classifiers[0].predict(X_test)

            y_true_binary = (np.array(labels) == target_label).astype(int)
            y_pred_binary = (np.array(y_pred) == target_label).astype(int)
            if score == "f1":
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                scores.append(f1)
            else:
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                scores.append(recall)

    return scores

def per_classifier_label_f1_multi(args, classifiers, target_label, all_X_test, labels,score ='accuracy'):
    scores = []
    for idx,clf in enumerate(classifiers):
        num_layer, num_head = flattened_idx_to_layer_head(idx, args.heads_num)
        # _,X_test =  get_head_weight(train_data,test_data, head_num = idx)
        X_test = all_X_test[:, num_layer, num_head, :]  # Select the features for the current classifier
        y_pred = clf.predict(X_test)

        # y_true_binary = (np.array(labels) == target_label).astype(int)
        # y_pred_binary = (np.array(y_pred) == target_label).astype(int)
        y_true_binary = np.array(labels)
        y_pred_binary = np.array(y_pred)
        if score == "f1":
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            scores.append(f1)
        else:
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            scores.append(accuracy)

    return scores

def main(args):
    train_data, test_data, train_labels, test_labels, train_pos, test_pos, topk_train, topk_test = load_data(args.model, args)
    if args.first_four:
        train_data, test_data, train_labels, test_labels, train_pos, test_pos, topk_train, topk_test = filter_data_first_four(train_data, test_data, train_labels, test_labels, train_pos, test_pos, topk_train, topk_test)
    unique_elements, counts = np.unique(train_labels, return_counts=True)
    count_dict = dict(zip(unique_elements, counts))
    print("train count: ", count_dict)
    unique_elements, counts = np.unique(test_labels, return_counts=True)
    count_dict = dict(zip(unique_elements, counts))
    print("test count: ", count_dict)
    empty_indices = [i for i, item in enumerate(train_pos) if item == []]
    if empty_indices:
        print(f"Empty indices in train_pos: {empty_indices}")
        train_data = [i for j, i in enumerate(train_data) if j not in empty_indices]
        train_labels = np.delete(train_labels, empty_indices)
        train_pos = [i for j, i in enumerate(train_pos) if j not in empty_indices]
        topk_train = [i for j, i in enumerate(topk_train) if j not in empty_indices]
    encoder = LabelEncoder()
    
    y_train = torch.tensor(encoder.fit_transform(train_labels), dtype=torch.long)
    y_test = torch.tensor(encoder.transform(test_labels), dtype=torch.long)
    label_mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
    label_names = []
    for i in label_mapping:
        label_names.append(i)
    # for i in label_names:
    #     print(f"Label {i} has index {label_mapping[i]}")
    layer_bias = args.layer_bias
    for bias in [layer_bias]:
        pos = args.pos
        print(f"\nRunning with BIAS={bias}, POSITION={pos}")
        X_train, X_test = get_head_weight(train_data, test_data, args.layer_num, args.heads_num, args.dim,
                                            pos, -1, bias, train_pos, test_pos, topk_train, topk_test)
        input_dim = args.dim * 2 if bias else args.dim
        #print(X_test)
        all_X_train = torch.tensor(X_train, dtype=torch.float32)
        all_X_test = torch.tensor(X_test, dtype=torch.float32)
        
        # train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1, shuffle=False)
        # val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_X_train = rearrange(all_X_train, 'b (l h) d -> b l h d', l=args.layer_num, h=args.heads_num).numpy()
    all_X_test = rearrange(all_X_test, 'b (l h) d -> b l h d', l=args.layer_num, h=args.heads_num).numpy()
    #all_head_accs = []
    probes = []
    if args.multi_classifier:

        f1_scores = {}
        for i in tqdm(range(len(label_names)), desc="Calculating F1 scores"):
            # X_test = all_X_test[, :]
            pos_idx = (y_train == i).nonzero(as_tuple=True)[0]
            neg_idx = (y_train != i).nonzero(as_tuple=True)[0]

            # 2️⃣ 随机采样负样本索引（数量等于正样本）
            num_pos = len(pos_idx)
            sampled_neg_idx = neg_idx[torch.randperm(len(neg_idx))[:num_pos]]

            # 3️⃣ 合并索引 & 打乱顺序
            combined_idx = torch.cat([pos_idx, sampled_neg_idx])
            perm = torch.randperm(len(combined_idx))
            shuffled_idx = combined_idx[perm]

            # 4️⃣ 过滤训练集
            filtered_X_train = all_X_train[shuffled_idx]
            filtered_y_train = torch.cat([
                torch.ones(len(pos_idx), dtype=torch.long),
                torch.zeros(len(sampled_neg_idx), dtype=torch.long)
            ])[perm]  # 保证 y 和 X 对齐

            # 5️⃣ 过滤测试集 (也变成二分类)
            test_pos_idx = (y_test == i).nonzero(as_tuple=True)[0]
            test_neg_idx = (y_test != i).nonzero(as_tuple=True)[0]
            #sampled_test_neg_idx = test_neg_idx[torch.randperm(len(test_neg_idx))[:len(test_pos_idx)]]

            #combined_test_idx = torch.cat([test_pos_idx, sampled_test_neg_idx])
            perm = torch.randperm(len(test_pos_idx))
            shuffled_test_idx = test_pos_idx[perm]

            filtered_X_test = all_X_test[shuffled_test_idx]
            filtered_y_test = torch.cat([
                torch.ones(len(test_pos_idx), dtype=torch.long),
            ])[perm]
            print(f"Training size for {label_names[i]} is {len(filtered_X_train)}")
            print(filtered_y_train)
            print(f"Testing size for {label_names[i]} is {len(filtered_X_test)}")
            print(filtered_y_test)
            probes = train_probes_multi(args, filtered_X_train, filtered_y_train)
            f1_scores[i] = np.array(per_classifier_label_f1_multi(args, probes, i, all_X_test=filtered_X_test, labels=filtered_y_test.numpy()))
    else:

        probes = train_probes_single(args, all_X_train, y_train)

        f1_scores = {}
        for i in tqdm(range(len(label_names)), desc="Calculating F1 scores"):
            # X_test = all_X_test[, :]
            f1_scores[i] = np.array(per_classifier_label_f1_single(args, probes, i, train_data=train_data, all_X_test=all_X_test, labels=y_test.numpy()))

    sorted_indices = {}
    scores_save = {}
    for label in f1_scores:
        scores_save[label_names[label]] = f1_scores[label].tolist()
        importances = f1_scores[label]
        #print(importances)
        top_indices = np.argsort(np.array(importances))[::-1]
        label_name = label_names[label]
        sorted_indices[label_name] = top_indices.tolist()

        # print(f"Label {label_name} sorted top features:")
        # print(top_indices[:10])  
        # print("-" * 30)
    #print(scores_save)
    if not os.path.exists(f"./main_results/{args.model}"):
        os.makedirs(f"./main_results/{args.model}")
    with open(f"./main_results/{args.model}/importance_{args.model}_{args.pos}_indices_{args.layer_bias}.json", 'w') as f:
        json.dump(sorted_indices, f)
    with open(f"./main_results/{args.model}/importance_{args.model}_{args.pos}_scores_{args.layer_bias}.json", 'w') as f:
        json.dump(scores_save, f)
    print("done")


    # top_heads = []

    # top_accs = np.argsort(all_head_accs_np.reshape(args.heads_num*args.layers_num))[::-1]
    # top_heads = [flattened_idx_to_layer_head(idx, args.heads_num) for idx in top_accs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--layer_num', type=int, default=36)
    parser.add_argument('--heads_num', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--pos', type=str, default="topk")
    parser.add_argument('--all_train', type=bool, default=False)
    parser.add_argument('--multi_classifier', type=bool, default=True)
    parser.add_argument('--layer_bias', type=bool, default=True)
    parser.add_argument('--first_four', type=bool, default=False)
    args = parser.parse_args()
    
    with open(f'./model_config.json', "r") as f:
        model_config = json.load(f)
    for config in model_config:
        if config["model_name"] == args.model:
            args.layer_num = config["layer_num"]
            args.heads_num = config["heads_num"]
            args.dim = config["dim"]
            break
    main(args)
