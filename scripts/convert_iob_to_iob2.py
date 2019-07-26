
def convert_iob_to_iob2(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        input_lines = f.readlines()
    with open(output_path, "w", encoding="utf-8") as f:
        chunk = False
        for line in input_lines:
            tags = line.split()
            ner_tag = tags[-1] if tags else None
            if ner_tag and ner_tag.startswith("B-"):
                chunk = True
            elif ner_tag and ner_tag.startswith("I-"):
                if not chunk:
                    tags[-1] = ner_tag.replace("I-", "B-")
                    chunk = True
            else:
                chunk = False
            f.write(" ".join(tags) + "\n")
