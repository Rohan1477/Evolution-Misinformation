import ast
import random

def is_content_in_file(content, filename):
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    post = ast.literal_eval(line)
                    if "content" in post and post["content"] == content:
                        return True
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping invalid line in {filename}: {line}")
    except FileNotFoundError:
        return False
    return False

with open("predicted_unrelated.txt", "r") as f:
    lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        try:
            line = ast.literal_eval(line.replace("\\/", "/"))
        except (ValueError, SyntaxError) as e:
            print(f"Skipping invalid line in predicted_false.txt: {line}")
            continue
        
        if "http" in line["content"] or "evolution of" in line["content"] or "EvolutionHasNoPorpoise" in line["content"] or "humanities" in line["content"]:
            continue
        
        if is_content_in_file(line["content"], "datasets/catagorized_test_data.txt"):
            continue
        
        print(line["content"])
        i = input()
        
        with open("datasets/catagorized_test_data.txt", "a") as f3:
            f3.write(str(line) + "\n")
        
        if i == "t":
            with open("datasets/test_data_evolution_true.txt", "a") as f2:
                f2.write(str(line) + "\n")
        elif i == "f":
            with open("datasets/test_data_evolution_false.txt", "a") as f2:
                f2.write(str(line) + "\n")
        elif i == "u":
            with open("datasets/test_data_not_evolution.txt", "a") as f2:
                f2.write(str(line) + "\n")
        elif i == "o":
            with open("datasets/test_data_evolution_opinion.txt", "a") as f2:
                f2.write(str(line) + "\n")
        elif i == "q":
            break

        # true is anything that is related to evolution and should not have warning under it
        # false is anything that should have warning under it about being misinformation
