import re
import logging
from atomqa.client import call_gpt
import json

def extract_json(markdown_text):
    # match the value in ```json{code}```
    pattern = r'```json\s*\n?(.*?)```'
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    if len(matches) == 0:
        pattern = r'```\n(.*?)\n```'
        matches = re.findall(pattern, markdown_text, re.DOTALL)

    return matches

def extract_solution(markdown_text):
    # match the value in <information>...</information>
    pattern = r'<information>\s*(.*?)\s*</information>'
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    return matches

def find_missing_keys(ground_truth: dict, input_dict: dict, path: str = "") -> list:
    """
    递归地查找 input_dict 相对于 ground_truth 缺失的键，并返回它们的路径。

    Args:
        ground_truth: 基准字典。
        input_dict: 需要检查的字典。
        path: 内部使用，用于追踪当前递归的路径。

    Returns:
        一个包含所有缺失键路径字符串的列表。
    """
    missing = []

    # 遍历基准字典
    if not isinstance(input_dict, dict):
        raise ValueError("Expected a JSON object")
    for key in ground_truth:
        # 构建当前键的路径
        current_path = f"{path}.{key}" if path else key

        # 1. 检查键是否存在
        if key not in input_dict:
            missing.append(current_path)
        else:
            # 2. 如果键存在且值为字典，则递归查找
            gt_value = ground_truth[key]
            if isinstance(gt_value, dict):
                # 将下层找到的缺失键合并到当前结果中
                missing.extend(find_missing_keys(gt_value, input_dict[key], path=current_path))

    return missing

def get_gpt_output(sys_prompt, user_prompt, error_fix_prompt, desired_outpot_dict):
    """
    Call GPT with the given system prompt and user prompt.
    """
    all_design_traj = []
    all_design_traj.append({"role": "system", "content": sys_prompt})
    all_design_traj.append({"role": "user", "content": user_prompt})
    logging.info(f"Calling GPT with prompt: \n{user_prompt}")

    max_it = 10
    iteration = 0
    test_json = None
    while iteration < max_it:
        response = call_gpt(all_design_traj, model="o3-mini")
        if response is None or len(response) == 0:
            logging.error("GPT response is None or empty.")
            iteration += 1
            continue
        response_text = response[0]
        if response_text is None:
            logging.error("GPT response text is None.")
            iteration += 1
            continue
        all_design_traj.append({"role": "assistant", "content": response_text})
        logging.info(f"GPT response: {response_text}")
        try:
            test_code = extract_json(response_text.strip())
            if len(test_code) == 0:
                logging.error("The extracted code is empty.")
                reason = "The output is not wrapped in a proper Json code block using triple backticks."
                feedback = error_fix_prompt.format(error_reason=reason)
                all_design_traj.append({"role": "user", "content": feedback})
                iteration += 1
                continue
            else:
                test_json = json.loads(test_code[0].replace("\n", "").replace("\t", ""))
        except Exception as e:
            logging.error(f"Error extracting code from model output: {e}")
            reason = f"Error extracting json from output: {e}"
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue

        if test_json is None:
            logging.error("The extracted json is None.")
            reason = "The extracted json is None."
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue

        try:
            missing_keys = find_missing_keys(desired_outpot_dict, test_json)
        except Exception as e:
            logging.error(f"Error checking missing keys: {e}")
            reason = f"Error checking missing keys: {e}"
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue

        if len(missing_keys) > 0:
            logging.error(f"Missing keys in the output: {missing_keys}")
            reason = f"Missing keys in the output: {missing_keys}"
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue
        else:
            return test_json

    return None
