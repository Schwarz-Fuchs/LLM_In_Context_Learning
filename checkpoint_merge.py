import json
import copy


def merge_b_into_a(data_a, data_b):
    """
    递归将 data_b 中存在但 data_a 中缺失的数据增量合并入 data_a 的副本中。
    不会修改原始的 data_a 和 data_b。
    """
    # 深拷贝 A，确保原数据不变
    result = copy.deepcopy(data_a)

    def _recursive_merge(target, source):
        for key, value in source.items():
            if key not in target:
                # 如果 target 中缺少该键，直接将 source 的值深拷贝加入
                target[key] = copy.deepcopy(value)
            elif isinstance(target[key], dict) and isinstance(value, dict):
                # 如果双方都是字典，继续递归深层对比合并
                _recursive_merge(target[key], value)
            # 如果 target[key] 已存在且不是字典，则保留 A 的原始数据，不做覆盖

    _recursive_merge(result, data_b)
    return result


def merge_json_files(file_a_path, file_b_path, output_path):
    # 读取原始 JSON 文件
    with open(file_a_path, 'r', encoding='utf-8') as f:
        json_a = json.load(f)

    with open(file_b_path, 'r', encoding='utf-8') as f:
        json_b = json.load(f)

    # 执行增量合并
    merged_data = merge_b_into_a(json_a, json_b)

    # 将结果写入新的 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"合并完成！已生成新文件: {output_path}")


# ================= 示例调用 =================
if __name__ == "__main__":
    # 替换为你的文件路径
    path_a = "benchmark_checkpoint_bk261008.json"
    path_b = "benchmark_checkpoint_ds.json"
    path_output = "benchmark_checkpoint_bk261008_merge.json"

    merge_json_files(path_a, path_b, path_output)