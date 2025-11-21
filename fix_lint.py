from collections import defaultdict
import shutil
import os
import re
import subprocess
from functools import partial
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("files", nargs="*", help="files to perform corrections. All if unspecified.")
args = parser.parse_args()


def gen_nindent(line):
    return len(re.search("^ *", line).group()) // 4


def C819(old_line: str, char_num: int, **kwargs):
    line = old_line[::-1]
    char_num = len(line) - char_num
    idx = line.find(",", char_num)
    line = line[:idx] + line[idx + 1:]
    return line[::-1]


C818 = C819


def Q000(old_line: str, char_num: int, **kwargs):
    if old_line[char_num - 1] != "'":
        if old_line[char_num - 1] == '"':
            return
        elif old_line[char_num] == "'":
            char_num += 1
    second_quote = old_line.find("'", char_num)
    if second_quote == -1:
        return
    return old_line[:char_num - 1] + '"' + old_line[char_num:second_quote] + '"' + old_line[second_quote + 1:]


def C812(old_line: str, char_num: int, **kwargs):
    return (old_line[:char_num - 1] + ",  " + old_line[char_num - 1:-1].lstrip(" ")).rstrip(" ") + "\n"


C815 = C812


def E261(old_line: str, char_num: int, **kwargs):
    i = old_line.find("#")
    return old_line[:i].rstrip(" ") + "  " + old_line[i:]


def W605(old_line: str, char_num: int, **kwargs):
    return old_line[:char_num] + "\\" + old_line[char_num:]


def E252(old_line: str, char_num: int, **kwargs):
    if not (char := old_line[char_num - 1:].lstrip(" ")[0]) in ["=", "+", "-", "**", "*", "/"]:
        return
    char_num += old_line[char_num - 1:].find(char) - 1
    if char == "=" and old_line[char_num] == "=":
        char = "=="
    return old_line[:char_num].rstrip(" ") + f" {char} " + old_line[char_num + len(char):].lstrip(" ")


E225 = E252
E221 = E252


def W503(old_line: str, char_num: int, **kwargs):
    if char_num == -1:
        return old_line.rstrip("\n ")
    idx1 = old_line[char_num:].find(" ")
    return " " + old_line[char_num - 1:char_num + idx1] + "\n" + \
        old_line[:char_num - 1] + old_line[char_num + idx1:].lstrip(" ")


def E127(old_line: str, char_num: int, **kwargs):
    if old_line[:4] == "    ":
        return old_line[4:]


def E303(old_line: str, char_num: int, **kwargs):
    return ""


W391 = E303


def E701(old_line: str, char_num: int, **kwargs):
    tabs = old_line[:char_num].count("    ")
    if old_line[char_num - 1] == ":":
        tabs += 1
    return old_line[:char_num] + "\n" + "    " * tabs + old_line[char_num:].lstrip(" ")


def W293(old_line: str, char_num: int, **kwargs):
    return "\n"


def E203(old_line: str, char_num: int, **kwargs):
    return old_line[:char_num].rstrip(" ") + old_line[char_num:]


E502 = E203
E201 = E203
E202 = E203


def F541(old_line: str, char_num: int, **kwargs):
    return old_line[:char_num - 1] + old_line[char_num:]


def W292(old_line: str, char_num: int, add_new_lines=1, **kwargs):
    return old_line + "\n" * add_new_lines


def E302(old_line: str, char_num: int, description: str, **kwargs):
    lines_expected, lines_found = re.findall("expected (\d*) blank line.*?, found (\d*)", description)[0]
    return (int(lines_expected) - int(lines_found)) * "\n" + old_line


E305 = E302
E301 = E302


def F401(old_line: str, char_num: int, function: str, ignore=False, **kwargs):
    if ignore:
        return old_line[:-1] + "  # noqa\n"
    *module, function = function.split(".")
    module = ".".join(module)
    if match := re.findall("from (.*?) import (.*?) *$", old_line):
        line_module, line_functions = match[0]
        line_functions_split = re.split(" *, *", line_functions)
        if line_module != module:
            return
        if function not in line_functions_split:
            return
        line_functions_split.remove(function)
        if not line_functions_split:
            return ""
        return old_line.replace(line_functions, ", ".join(line_functions_split))
    elif old_line.startswith("import"):
        module = f"{module}.{function}" if module else function
        if re.findall("import (.*?) *$", old_line)[0] == module:
            return ""


def E265(old_line: str, char_num: int, symbol="#", **kwargs):
    return re.sub(symbol + "+ *", symbol + " ", old_line)


E241 = partial(E265, symbol=",")


def F841(old_line: str, char_num: int, **kwargs):
    end_var_def = old_line.find("=", char_num) + 1
    return old_line[:char_num - 1] + old_line[end_var_def:].lstrip(" ")


def E122(old_line: str, char_num: int, add=1, **kwargs):
    return (gen_nindent(old_line) + add) * "    " + old_line.lstrip(" ").rstrip(" ")


E123 = E126 = partial(E122, add=-1)
E131 = E122
E111 = E122
E121 = E122


def W291(old_line: str, char_num:int, **kwargs):
    return old_line[:-1].rstrip(" ") + "\n"


def E262(old_line: str, char_num:int, **kwargs):
    return re.sub(" *#+ *", "  # ", old_line)


def E231(old_line: str, char_num: int, **kwargs):
    if old_line[char_num-1] != ",":
        return
    return old_line[:char_num] + " " + old_line[char_num:].lstrip(" ")


def E241(old_line: str, char_num: int, **kwargs):
    return old_line[:char_num].rstrip(" ") + " " + old_line[char_num:].lstrip(" ")


E272 = E241
E222 = E241


def E226(old_line: str, char_num: int, **kwargs):
    op = old_line[char_num - 1]
    if op not in "+/*-|&":
        return
    return old_line[:char_num - 1].rstrip(" ") + f" {op} " + old_line[char_num:].lstrip(" ")


def E251(old_line: str, char_num: int, **kwargs):
    if old_line[char_num - 1:].lstrip(" ")[0] == "=":
        return old_line[:char_num].rstrip(" ") +  old_line[char_num - 1:].lstrip(" ")
    elif old_line[:char_num].rstrip(" ")[-1] == "=":
        return old_line[:char_num].rstrip(" ") + old_line[char_num:].lstrip(" ")


linting = subprocess.run(["./tests/run_all"], capture_output=True)
linting_out = linting.stdout.decode().strip("\n")
with open("lint.txt", "w") as doc:
    doc.write(linting_out)
linting_out = linting_out.split("\n")
print(linting_out[0])
if not (err := linting.stderr.decode()):
    print("all fine!")
    exit()
print(err)

errors = defaultdict(lambda: defaultdict(dict))
for line in linting_out[1:]:
    file, code, description = line.split(" ", maxsplit=2)
    file, line_num, char_num = file.rstrip(":").split(":")
    if args.files and file not in args.files:
        continue
    output = [code, {"description": description}]
    if code == "W503":
        errors[file][int(line_num) - 1][-1] = output
    elif code == "E303":
        num = int(re.search("\(\d*\)", description).group()[1:-1])
        for i in range(num):
            errors[file][int(line_num) - 1 - i][-1] = output
        continue
    elif code == "F401":
        if "__init__" in file:
            output[-1]["ignore"] = True
        function = re.findall("'(.*?)'", description)[0]
        output[-1]["function"] = function
        char_num = len(errors[file][int(line_num)])
    errors[file][int(line_num)][int(char_num)] = output

to_commit = []
for file in errors:
    with open(file) as doc:
        lines = list(doc.readlines())

    changed = False
    for line_num in errors[file]:
        old_line = lines[line_num - 1]
        new_line = old_line
        line_errs = []
        for char_num in errors[file][line_num]:
            error, kwargs = errors[file][line_num][char_num]
            if error not in ["C819", "Q000", "C812", "E261", "W605", "E252", "E203", "E251",
                             "F541", "W291", "W292", "E265", "E305", "E122", "W503", "E111",
                             "E127", "E701", "E303", "E225", "E502", "W391", "W293", "E301",
                             "C815", "F401", "F841", "E302", "E241", "E126", "E131", "E226",
                             "E262", "E231", "E241", "E272", "E222", "E123", "E201", "E202",
                             "C818", "E121", "E221"]:
                continue
            if error == "E131":
                nindent_this = gen_nindent(new_line)
                prev_align_idx = 1
                while (nindent_prev := gen_nindent(lines[line_num - 1 - prev_align_idx])) == nindent_this:
                    prev_align_idx += 1
                kwargs["add"] = nindent_prev - nindent_this
            if (new_new_line := eval(error)(new_line, char_num, **kwargs)) is None:
                print(f"{error} on char {char_num} on line {line_num} not found in {file}")
                break
            new_line = new_new_line
            line_errs.append(error)
        if line_errs:
            changed = True
            lines[line_num - 1] = new_line
            print(f"fixing {', '.join(line_errs)} on line {line_num} in {file}:", kwargs["description"])
            print("--", old_line.replace('\n', '\\n'))
            print("++", lines[line_num - 1].replace('\n', '\\n'))
    if changed and input("write changes? (y/n)") == "y":
        shutil.copy(file, file + '.tmp')
        with open(file, "w") as doc:
            doc.writelines(lines)
        os.remove(file + '.tmp')
        to_commit.append(file)

if to_commit and input("commit changes? (y/n). Beware the entire files are commited, including your own changes if any!") == "y":
    command = ["git", "commit", *to_commit]
    amend = input("amend? (y/n, default: n)")
    if amend == "y":
        command.append("--amend")
    default = "fix linting"
    message = input(f"commit message (default '{default}')")
    command.extend(["-m", message if message else default])
    commit = subprocess.run(command, capture_output=True)
    if err := commit.stderr.decode().strip("\n"):
        print(err)
    print(commit.stdout.decode().strip("\n"))
